# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:17:00 2025

@author: Chris
"""
###
"""
    This script contains functions for downloading Temperature (kelvin) data 
    from the NWS NDFD. The agency updates temperature forecast data over 
    the continental United States every 6 hours. Here, it is prepocessed, 
    converted to xarray format for manipulation, and used to generate a 
    .gif of the 7 day forecast.
    
"""
###

import os
import gc
import requests
import xarray as xr
import pandas as pd
import tempfile
import time
import ssl
import logging
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

def safe_download(url, max_retries=3):
    for attempt in range(max_retries):
        try:
            return download_and_process_grib(url)
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2 * attempt)
    return None

def download_and_process_grib(url):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # creating a temporary file 
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()  # Ensure data is written before access
            
            ds = xr.open_dataset(tmp.name, engine="cfgrib",
                                   backend_kwargs={"indexpath": ""},
                                   decode_timedelta='CFTimedeltaCoder')['t2m']
        return ds
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    finally:
        os.remove(tmp.name)
    
def safe_upload(supabase, bucket_name, supabase_path, local_file_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(local_file_path, 'rb') as f:
                supabase.storage.from_(bucket_name).upload( 
                    supabase_path,
                    f,
                    file_options={"content-type": "application/octet-stream",
                                  "upsert": "true"
                                  })
            del f
            gc.collect() # garbage collector
            return True
        except ssl.SSLError as ssl_err:
            logger.error(f"SSL error on attempt {attempt+1}: {ssl_err}")
            time.sleep(2* (attempt+1))
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {e}")
            break
    return False



def get_temperature():
    # URLs from NOAA NWS NDFD for grib files
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.temp.bin"
   
    logger.info("Downloading Latest NWS NDFD Temperature data...")
    
    ds_1thru3 = safe_download(url_days_1thru3)
    ds_4thru7 = safe_download(url_days_4thru7)
    
    logger.info("Preprocessing data...")
    
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
    
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("valid_time")
    
    # Adding Celsius and Farenheit values as coordinates
    combined_ds = combined_ds.assign_coords({"fahrenheit":(["step", "y","x"],
                                                           (((combined_ds.values-273.15)*1.8)+32))})
    combined_ds = combined_ds.assign_coords({"celsius":(["step", "y","x"],
                                                        (combined_ds.values-273.15))})

    logger.info("Saving Resultant Dataset to Cloud...")
    # Cloud Access
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    bucket_name = "maps"
    storage_path_prefix = "processed-data/Temp_Latest.zarr"
       
    # Initialize SupaBase Bucket Connection
    supabase: Client = create_client(database_url, api_key)
    
    # write ds to temporary directory
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/mydata.zarr"
            # save as scalable chunked cloud-optimized zarr file
            combined_ds.to_zarr(zarr_path, mode="w", consolidated=True)
        
            # recursively save zarr directories
            for root, dirs, files in os.walk(zarr_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                
                    # Convert local path to relative path for Supabase
                    relative_path = os.path.relpath(local_file_path, zarr_path)
                    supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"
                    
                    uploaded = safe_upload(supabase, bucket_name, supabase_path, local_file_path)
                    if not uploaded:
                        logger.error(f"Final failure for {relative_path}")
                    gc.collect()
        
        logger.info('Latest 6-hourly 7-Day Forecast Saved to Cloud!')
        return combined_ds
    except:
        logger.error("Saving final dataset failed")
    gc.collect() # garbage collector. deletes objects that are no longer in use
                    
    

