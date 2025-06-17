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
import psutil
import requests
import xarray as xr
import pandas as pd
import tempfile
import time
import logging
import httpx
from mimetypes import guess_type
from pathlib import Path

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
            temp_file_path = tmp.name
            
            ds = xr.open_dataset(temp_file_path, engine="cfgrib",
                                   backend_kwargs={"indexpath": ""},
                                   decode_timedelta='CFTimedeltaCoder')['t2m'].load()
        os.remove(temp_file_path)
        return ds
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

    
def upload_zarr_to_supabase(
    url: str,
    api_key: str,
    bucket_name: str,
    local_zarr_path: str,
    remote_prefix: str,
    overwrite: bool = True
):
    """
    Recursively uploads all files from local Zarr directory to Supabase
    using direct HTTP requests

    Parameters:
        supabase_url (str): Supabase project URL.
        supabase_key (str): Supabase service role key.
        bucket_name (str): Name of the storage bucket.
        local_zarr_path (str): Path to the local `.zarr` folder.
        remote_prefix (str): Remote path in the bucket.
        overwrite (bool): Whether to overwrite existing files. Default is True.
    
    """

    local_path = Path(local_zarr_path).resolve()
    if not local_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {local_path}")

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true" if overwrite else "false"
    }

    logger.info(f"Uploading contents of {local_path} to Supabase bucket '{bucket_name}' at '{remote_prefix}/'")

    with httpx.Client(timeout=60) as client:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                supabase_path = f"{remote_prefix}/{relative_path.replace(os.sep, '/')}"

                mime_type, _ = guess_type(file)
                mime_type = mime_type or "application/octet-stream"

                with open(local_file_path, "rb") as f:
                    response = client.put(
                        f"{url}/storage/v1/object/{bucket_name}/{supabase_path}",
                        content=f.read(),
                        headers={**headers, "Content-Type": mime_type}
                    )

                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Uploaded: {supabase_path}")
                    else:
                        logger.error(f"❌ Upload failed for {supabase_path}: {response.status_code} - {response.text}")
                gc.collect()

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
    
    # Saving ds to cloud 
    log_memory_usage("Before recursively saving zarr to Cloud")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # save ds to temporary file
            zarr_path = f"{tmpdir}/mydata.zarr"
            # save as scalable chunked cloud-optimized zarr file
            combined_ds.to_zarr(zarr_path, mode="w", consolidated=True)
            # recursively uploading zarr data
            upload_zarr_to_supabase(
                url = "https://rndqicxdlisfpxfeoeer.supabase.co",
                api_key = os.environ['SUPABASE_KEY'],
                bucket_name = "maps",
                local_zarr_path = zarr_path,
                remote_prefix = "processed-data/Temp_Latest.zarr"
            )
            logger.info('Latest 6-hourly 7-Day Forecast Saved to Cloud!')
            log_memory_usage("After recursively saving zarr to cloud")
            return combined_ds
    except:
        logger.error("Saving final dataset failed")