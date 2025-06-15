# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:24:08 2025

@author: Chris
"""
###
"""
    This script contains functions for downloading Sky Coverage (%) data 
    from the NWS NDFD. The agency updates sky coverage forecast data over 
    the continental United States every 6 hours. Here, it is prepocessed, 
    converted to xarray format for manipulation, and used to generate a 
    .gif of the 7 day forecast.
    
"""
###

import os
import gc
import psutil
import requests
import httpx
import xarray as xr
import pandas as pd
import tempfile
import time
import logging

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
        # Use a context manager to ensure connection closes
        with requests.get(url, timeout=60, stream=True) as response:
            response.raise_for_status()
            content = response.content  # This loads the full response into memory

        # Now work with the content after closing the connection
        with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
            tmp.write(content)
            tmp.flush()
            temp_file_path = tmp.name

        ds = xr.open_dataset(temp_file_path, engine="cfgrib",
                             backend_kwargs={"indexpath": ""},
                             decode_timedelta='CFTimedeltaCoder')['unknown'].load()
        os.remove(temp_file_path)
        return ds
    except Exception as e:
        logger.error(f"Error: {e}")
        return None
    
def upload_to_supabase_with_client(client, bucket, path, local_file_path):
    headers = {
        "apikey": os.environ["SUPABASE_KEY"],
        "Authorization": f"Bearer {os.environ['SUPABASE_KEY']}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true"
    }

    with open(local_file_path, "rb") as f:
        url = f"https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/{bucket}/{path}"
        response = client.put(url, content=f, headers=headers)
        if response.status_code in [200, 201]:
            return True
        else:
            logger.warning(f"Upload failed: {response.status_code} - {response.text}")
            return False

def get_sky_coverage():    
    # URLs from NOAA NWS NDFD and grib cloud paths
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.sky.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.sky.bin"
    
    logger.info('Downloading latest Sky Coverage dataset from NWS NDFD...')
    
    ds_1thru3 = safe_download(url_days_1thru3)
    ds_4thru7 = safe_download(url_days_4thru7)
    
    logger.info('Preprocessing Data...')
    
    # Getting 6-hourly data across all days
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
    
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    gc.collect()
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("valid_time")

    # Creating zarr file storage path to storage bucket
    logger.info("Saving Resultant Dataset to Cloud...")
    bucket_name = "maps"
    storage_path_prefix = "processed-data/SkyCover_Latest.zarr"
    
    # Saving ds to cloud 
    log_memory_usage("Before recursively saving zarr to Cloud")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # save ds to temporary file
            zarr_path = f"{tmpdir}/mydata.zarr"
            # save as scalable chunked cloud-optimized zarr file
            combined_ds.to_zarr(zarr_path, mode="w", consolidated=True)
            # recursively uploading zarr data
            with httpx.Client() as client:
                for root, dirs, files in os.walk(zarr_path):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file_path, zarr_path)
                        supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"
                        uploaded = upload_to_supabase_with_client(client, bucket_name, supabase_path, local_file_path)
                    if not uploaded:
                        logger.error(f"Final failure for {relative_path}")
                    gc.collect()
    
        logger.info('Latest 6-hourly 7-Day Forecast Saved to Cloud!')
        log_memory_usage("After recursively saving zarr to cloud")
        gc.collect()
        return combined_ds
    except:
        logger.error("Saving final dataset failed")
    
    
    