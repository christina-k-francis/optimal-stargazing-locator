# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:51:03 2025

@author: Chris
"""
###
"""
    This script downloads Precipitation Probability data from the NWS NDFD. 
    The agency updates sky coverage forecast data over the continental 
    United States every 6 hours. Here, it is prepocessed, converted to xarray
    format for manipulation, and used to generate a .gif of the 7 day forecast.
    
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
import ssl
import logging
from supabase import create_client, Client
from pathlib import Path
from mimetypes import guess_type

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
            temp_file_name = tmp.name
            
            ds = xr.open_dataset(temp_file_name, engine="cfgrib",
                                   backend_kwargs={"indexpath": ""},
                                   decode_timedelta='CFTimedeltaCoder')['unknown'].load()
        os.remove(temp_file_name)
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
    Recursively uploads all files from a local Zarr directory to Supabase Storage.

    Parameters:
        supabase_url (str): Supabase project URL.
        supabase_key (str): Supabase service role key.
        bucket_name (str): Name of the storage bucket.
        local_zarr_path (str): Path to the local `.zarr` folder.
        remote_prefix (str): Remote path in the bucket.
        overwrite (bool): Whether to overwrite existing files. Default is True.
    """
    client = create_client(url, api_key)
    storage = client.storage.from_(bucket_name)

    local_path = Path(local_zarr_path).resolve()

    if not local_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {local_path}")

    logger.info(f"Uploading contents of {local_path} to Supabase bucket '{bucket_name}' at '{remote_prefix}/'")

    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            remote_path = f"{remote_prefix}/{relative_path.as_posix()}"

            mime_type, _ = guess_type(file_path.name)
            mime_type = mime_type or "application/octet-stream"

            with open(file_path, "rb") as f:
                try:
                    storage.upload(remote_path, f, {"content-type": mime_type}, file_options={"upsert": overwrite})
                    logger.info(f"✅ Uploaded: {remote_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to upload {remote_path}: {e}")


def get_precip_probability():
    # URLs from NOAA NWS NDFD and grib cloud paths
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.pop12.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.pop12.bin" 
    
    logger.info("Downloading Latest Precipitation Probability data...")
    
    ds_1thru3 = safe_download(url_days_1thru3)
    ds_4thru7 = safe_download(url_days_4thru7)
    
    logger.info("Preprocessing data...")
    
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 12-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
       
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("valid_time")
       
    logger.info("Saving Resultant Dataset to Cloud...")
       
    # Cloud Access
    upload_zarr_to_database(
        url = "https://rndqicxdlisfpxfeoeer.supabase.co",
        api_key = os.environ['SUPABASE_KEY'],
        bucket_name = "maps",
        local_zarr_path = "PrecipProb_Latest.zarr"
        remote_prefix = "processed-data/PrecipProb_Latest.zarr"
    )
    
