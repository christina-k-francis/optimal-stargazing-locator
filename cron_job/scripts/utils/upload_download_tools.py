"""
Here are functions useful for uploading and downloading multidimensional 
xarray datasets/zarr files to and from supabase with fail-proof safety nets.
"""

import xarray as xr
import rioxarray
import os
import time
import httpx
import fsspec
import tempfile
import logging
import warnings
import mimetypes
import requests
import shutil
from storage3 import create_client

logger = logging.getLogger(__name__)
MAX_RETRIES = 5
DELAY_BETWEEN_RETRIES = 2  # seconds

# Redirect all warnings to the logger
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=UserWarning)

# silence packages with noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

def upload_zarr_dataset(nws_ds, storage_path_prefix: str, bucket_name="maps"):
    """
    Saves an xarray Dataset as Zarr and uploads it recursively to Supabase with retries.
    
    Parameters:
    - combined_ds (xr.Dataset): Dataset to upload
    - storage_path_prefix (str): Supabase key prefix (e.g., processed-data/Temp_Latest.zarr)
    - bucket_name (str): Supabase storage bucket (default "maps")
    """
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ.get("SUPABASE_KEY")
    if not api_key:
        raise EnvironmentError("SUPABASE_KEY is required but not set.")

    storage = create_client(f"{database_url}/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path = os.path.join(tmpdir, "data.zarr")
        logger.info("Writing dataset to Zarr...")
        nws_ds.load()  # force all lazy computations
        nws_ds.to_zarr(zarr_path, mode="w", consolidated=True)

        logger.info("Uploading Zarr dataset to Supabase...")
        for root, _, files in os.walk(zarr_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, zarr_path)
                supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"

                mime_type, _ = mimetypes.guess_type(file)
                if mime_type is None:
                    mime_type = "application/octet-stream"

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        with open(local_file_path, "rb") as f:
                            storage.from_(bucket_name).upload(
                                supabase_path,
                                f.read(),
                                {
                                    "content-type": mime_type,
                                    "x-upsert": "true"
                                }
                            )
                        break  # successful upload
                    except Exception as e:
                        logger.warning(f"Upload failed for {relative_path} (attempt {attempt}): {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(DELAY_BETWEEN_RETRIES)
                        else:
                            logger.error(f"Final failure uploading {relative_path}")
        logger.info("✅ Zarr dataset uploaded to Supabase successfully.")
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Tmpdir cleanup failed: {e}")


def download_grib_with_retries(url, variable_key, max_retries=5, timeout=90):
    """
    Downloads and processes a GRIB2 file from the given URL using cfgrib engine with retries.

    Parameters:
    - url (str): Remote URL to the GRIB2 file
    - variable_key (str): Key to select variable from dataset (e.g., "t2m", "tp")
    - max_retries (int): Maximum retry attempts
    - timeout (int): Timeout for HTTP GET in seconds

    Returns:
    - xarray.DataArray or None if all retries fail
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading GRIB2 file (Attempt {attempt}) from {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
                tmp.write(response.content)
                tmp.flush()
                temp_file_path = tmp.name

            # Parse with cfgrib
            ds = xr.open_dataset(
                temp_file_path,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
                decode_timedelta="CFTimedeltaCoder"
            )[variable_key].load()

            logger.info(f"✅ Successfully downloaded and parsed {variable_key}")
            os.remove(temp_file_path)
            return ds

        except Exception as e:
            logger.warning(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                logger.error("❌ All retry attempts failed.")
                return None
            wait_time = 2 ** attempt  # exponential backoff
            logger.info(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

def load_zarr_from_supabase(bucket, path):
    """
    Downloads the specified zarr file from the provided supabase storage path.

    Parameters:
    - bucket: supabase storage bucket name in string format
    - path: path within supabase storage bucket to the desired GeoTIFF in str format

    Returns:
    - xarray.DataArray 
    """
    url_base = f"https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/{bucket}/{path}/"
    fs = fsspec.filesystem("http")
    ds = xr.open_zarr(fs.get_mapper(url_base), consolidated=True,
                      decode_timedelta='CFTimedeltaCoder')
    return ds

def load_tiff_from_supabase(bucket: str, path: str):
    """
    Downloads the specified GeoTIFF from the provided supabase storage path.

    Parameters:
    - bucket: supabase storage bucket name in string format
    - path: path within supabase storage bucket to the desired GeoTIFF

    Returns:
    - xarray.DataArray or None
    """
    file_url = f"https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/{bucket}/{path}"
    with httpx.Client() as client:
        r = client.get(file_url)
        r.raise_for_status()
    
        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name  
                tmp.flush() # ensures data is written to disk
                
                da = rioxarray.open_rasterio(tmp_path, masked=True)
                os.remove(tmp.name) # ensure temp file is deleted
                return da
        except:
            logger.exception("geoTIFF download error")
