# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:58:55 2025

@author: Chris
"""
###
"""
    This script contains functions for downloading Wind Speed and
    Wind Direction data from the NWS NDFD. The agency updates the Wind
    Speed and Direction forecast data over the continental United States
    every 6 hours. Here, it is prepocessed, converted to xarray format 
    for manipulation, and used to generate a .gif of the 7 day forecast.
    
"""
###

import os
import gc
import psutil
import requests
import xarray as xr
import pandas as pd
import numpy as np
import tempfile
import time
import logging
import ssl
from mimetypes import guess_type
from storage3 import create_client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    """Logs the RAM usage (RSS Memory) at it's position in the script"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"[MEMORY] RSS memory usage {stage}: {mem:.2f} MB ")


def safe_download(url, max_retries=3):
    for attempt in range(max_retries):
        if 'wspd' in url:
            try:
                return download_and_process_speed_grib(url)
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 * attempt)
        elif 'wdir' in url:
            try:
                return download_and_process_dir_grib(url)
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                time.sleep(2 * attempt)
    return None

def download_and_process_dir_grib(url):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # creating a temporary file 
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()  # Ensure data is written before access
            temp_file_name = tmp.name
            
            ds = xr.open_dataset(temp_file_name, engine="cfgrib",
                                   backend_kwargs={"indexpath": ""},
                                   decode_timedelta='CFTimedeltaCoder')['wdir10'].load()
        os.remove(temp_file_name)
        return ds
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    
def download_and_process_speed_grib(url):
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # creating a temporary file 
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()  # Ensure data is written before access
            temp_file_name = tmp.name
            
            ds = xr.open_dataset(temp_file_name, engine="cfgrib",
                                   backend_kwargs={"indexpath": ""},
                                   decode_timedelta='CFTimedeltaCoder')['si10'].load()
        os.remove(temp_file_name)
        return ds
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    
def safe_upload(storage, bucket_name, supabase_path,
                local_file_path, file_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(local_file_path, 'rb') as f:
                storage.from_(bucket_name).upload( 
                    supabase_path,
                    f.read(),
                    file_options={"content-type": file_type,
                                  "upsert": "true"})
            return True
        except ssl.SSLError as ssl_err:
            logger.error(f"SSL error on attempt {attempt+1}: {ssl_err}")
            time.sleep(2* (attempt+1))
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {e}")
            break
    return False          

def get_wind_speed_direction():
    log_memory_usage("Before importing Wind Speed")
    # 1. STARTING WITH WIND SPEED
    # URLs from NOAA NWS NDFD for grib files
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wspd.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.wspd.bin"

    logger.info("Downloading Latest NWS NDFD Wind Speed data...")
    
    ds_1thru3 = safe_download(url_days_1thru3)
    ds_4thru7 = safe_download(url_days_4thru7)
    
    logger.info("Preprocessing data...")
    
    log_memory_usage("Before preprocess")
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
    
    # Merging speed datasets
    speed_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    speed_ds = speed_ds.sortby("valid_time")
    log_memory_usage("After preprocess, Before Wind Direction import")
    
    
    #2 NOW, LET'S RETRIEVE WIND DIRECTION
    # URLs from NOAA NWS NDFD for grib files
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wdir.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.wdir.bin"

    logger.info("Downloading Latest NWS NDFD Wind Direction data...")   

    ds_1thru3 = safe_download(url_days_1thru3)
    ds_4thru7 = safe_download(url_days_4thru7)

    logger.info('Preprocessing Data...')
     
    log_memory_usage("Before preprocess")
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)

    # Merging direction datasets
    direction_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    direction_ds = direction_ds.sortby("valid_time")
    gc.collect() # clean up!
    log_memory_usage("After preprocess")

    # Aggregating Wind Direction and Wind Speed into a single dataset
    logger.info('Merging Wind Speed and Wind Direction datasets...')
    log_memory_usage("Before masking datasets")
    masked_speed = np.ma.masked_invalid(speed_ds)
    masked_dir = np.ma.masked_invalid(direction_ds)
    log_memory_usage("After masking, before calculating u and v dir data")
    # Breaking direction into U and V components
    direction_coords = {
        'direction': (('step','y','x'), direction_ds.data),
        'U': (('step','y','x'), (-masked_speed * np.sin(np.deg2rad(masked_dir)))),
        'V': (('step','y','x'), (-masked_speed * np.cos(np.deg2rad(masked_dir))))
    }
    log_memory_usage("After u and v calc, before creating Wind DS")
    # adding wind direction to wind speed dataset
    speed_ds = speed_ds.assign_coords(direction_coords)
    log_memory_usage("After creating Wind DS")
    gc.collect() # cleaning up some more!
    
    logger.info("Saving Resultant Dataset to Cloud...")
    
    # Cloud Access
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    storage_path_prefix = "processed-data/Wind_Spd_Dir_Latest.zarr"
       
    # Initialize Supabase Storage Connection
    storage = create_client(f"{database_url}/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)
    
    # write ds to temporary directory
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zarr_path = f"{tmpdir}/mydata.zarr"
            # save as scalable chunked cloud-optimized zarr file
            speed_ds.to_zarr(zarr_path, mode="w", consolidated=True)
        
            # recursively save zarr directories
            for root, dirs, files in os.walk(zarr_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                
                    # Convert local path to relative path for Supabase
                    relative_path = os.path.relpath(local_file_path, zarr_path)
                    supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"
                    
                    mime_type, _ = guess_type(file)
                    if mime_type == None:
                        mime_type = "application/octet-stream"
                    
                    uploaded = safe_upload(storage,
                                           "maps",
                                           supabase_path, 
                                           local_file_path,
                                           mime_type)
                    if not uploaded:
                        logger.error(f"Final failure for {relative_path}")
            logger.info('Latest 6-hourly 7-Day Forecast Saved to Cloud!')
            return speed_ds
    except:
        logger.error("Saving final dataset failed")