# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:56:59 2025

@author: Chris
"""
import numpy as np
import xarray as xr
import os
import gc
import time
import tempfile
import httpx
import logging
from pathlib import Path
from mimetypes import guess_type

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def upload_zarr_to_supabase_raw(
    supabase_url: str,
    api_key: str,
    bucket: str,
    local_zarr_path: str,
    remote_prefix: str,
    overwrite: bool = True
):
    """Uploads all files in a local .zarr directory to Supabase Storage via HTTP PUT (robust method)."""
    local_path = Path(local_zarr_path).resolve()
    if not local_path.is_dir():
        raise ValueError(f"{local_zarr_path} is not a valid directory")

    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream",
        "x-upsert": "true" if overwrite else "false"
    }

    logger.info(f"Uploading contents of {local_path} to Supabase bucket '{bucket}' at '{remote_prefix}/'")

    with httpx.Client(timeout=60) as client:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_path)
                supabase_path = f"{remote_prefix}/{relative_path.replace(os.sep, '/')}"

                mime_type, _ = guess_type(file)
                mime_type = mime_type or "application/octet-stream"

                logger.info(f"Uploading file: {relative_path}")
                with open(local_file_path, "rb") as f:
                    response = client.put(
                        f"{supabase_url}/storage/v1/object/{bucket}/{supabase_path}",
                        content=f.read(),
                        headers={**headers, "Content-Type": mime_type}
                    )

                    if response.status_code in [200, 201]:
                        logger.info(f"✅ Uploaded: {supabase_path}")
                    else:
                        logger.error(f"❌ Upload failed for {supabase_path}: {response.status_code} - {response.text}")

                gc.collect()

np.random.sample(0)
lat = np.linspace(0, 75, 30)
lon = np.linspace(0, 100, 30)
temp = np.random.randint(0,100, (30,30))

temp_da = xr.DataArray( 
    data=temp,
    dims=["x","y"],
    coords={
        "x": lon,
        "y": lat})

# Cloud Access
supabase_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
api_key = os.environ['SUPABASE_KEY']

with tempfile.TemporaryDirectory() as tmpdir:
    zarr_path = f"{tmpdir}/test_dataset.zarr"
    
    logger.info("Saving dataset to Zarr...")
    temp_da.to_zarr(zarr_path, consolidated=True)
    
    # Safety sleep to ensure write is flushed on Render
    time.sleep(1)
    
    logger.info("Initiating upload to Supabase...")
    upload_zarr_to_supabase_raw(
        supabase_url=supabase_url,
        api_key=api_key,
        bucket="maps",
        local_zarr_path=zarr_path,
        remote_prefix="Test_Temp.zarr",
        overwrite=True
    )
    
    logger.info("✅ All files uploaded!")