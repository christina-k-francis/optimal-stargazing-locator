# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:56:59 2025

@author: Chris
"""
import numpy as np
import xarray as xr
import os
import time
import tempfile
import httpx
from mimetypes import guess_type
from supabase import create_client, Client

def upload_file_to_supabase_raw(supabase_url, api_key, bucket,
                                supabase_path, local_file_path,
                                file_type):
    with open(local_file_path, "rb") as f:
        headers = {
            "apikey": api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type": file_type,
            "x-upsert": "true"
        }

        response = httpx.put(
            f"{supabase_url}/storage/v1/object/{bucket}/{supabase_path}",
            content=f.read(),
            headers=headers
        )

        if response.status_code in [200, 201]:
            print(f"✅ Uploaded: {supabase_path}")
        else:
            print(f"❌ Upload failed for {supabase_path}: {response.status_code} - {response.text}") 

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
database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
api_key = os.environ['SUPABASE_KEY']
storage_path_prefix = "Test_Temp.zarr"
# Initialize SupaBase Bucket Connection
supabase: Client = create_client(database_url, api_key)

# write ds to temporary directory
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = f"{tmpdir}/mydata.zarr"
        # save as scalable chunked cloud-optimized zarr file
        temp_da.to_zarr(zarr_path, mode="w", consolidated=True)
        time.sleep(2) # Gives FS time to flush
    
        # recursively save zarr directories
        for root, dirs, files in os.walk(zarr_path):
            for file in files:
                if file.startswith(".") or not file.startswith("."):
                    _ = os.path.getsize(os.path.join(root, file))  # Access forces flush
                    local_file_path = os.path.join(root, file)
            
                    # Convert local path to relative path for Supabase
                    relative_path = os.path.relpath(local_file_path, zarr_path)
                    supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"
                
                    mime_type, _ = guess_type(file)
                    mime_type = mime_type or "application/octet-stream"
                
                    upload_file_to_supabase_raw(
                        supabase_url=database_url,
                        api_key=api_key,
                        bucket="maps",
                        supabase_path=supabase_path,
                        local_file_path=local_file_path,
                        file_type=mime_type
                    )
        print("Test Complete!")
except:
    print("upload failed")
