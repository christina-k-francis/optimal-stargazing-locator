"""
Here are functions useful for uploading and downloading multidimensional 
xarray datasets/zarr files to and from Cloudflare R2 with fail-proof safety nets.
"""

import xarray as xr
import rioxarray
import boto3
import os
import s3fs
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

def upload_zarr_dataset(nws_ds, storage_path_prefix: str,
                         bucket_name='optimal-stargazing-locator')
    """
    Saves an xarray Dataset as Zarr and uploads it recursively to Cloudflare R2 with retries.

    Parameters:
    - nws_ds (xr.Dataset): Dataset to upload
    - storage_path_prefix (str): R2 key prefix (e.g., processed-data/Temp_Latest.zarr)
    - bucket_name (str): R2 storage bucket (default "maps")
    """
    # Cloud Access Variables
    account_id = os.environ.get("R2_TOKEN")
    access_key = os.environ.get("R2_ACCESS_KEY")
    secret_key = os.environ.get("R2_SECRET_KEY")
    if not account_id or not access_key or not secret_key:
        raise EnvironmentError("R2 credentials (R2_TOKEN, R2_ACCESS_KEY, R2_SECRET_KEY) must be set.")

    # Endpoint format: https://<account_id>.r2.cloudflarestorage.com
    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    tmpdir = tempfile.mkdtemp()
    try:
        zarr_path = os.path.join(tmpdir, "data.zarr")
        logger.info("Writing dataset to Zarr...")
        nws_ds.to_zarr(zarr_path, mode="w", consolidated=True)

        logger.info("Uploading Zarr dataset to Cloudflare R2...")
        for root, _, files in os.walk(zarr_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, zarr_path)
                r2_key = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"

                mime_type, _ = mimetypes.guess_type(file)
                if mime_type is None:
                    mime_type = "application/octet-stream"

                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        with open(local_file_path, "rb") as f:
                            s3_client.upload_fileobj(
                                f,
                                bucket_name,
                                r2_key,
                                ExtraArgs={"ContentType": mime_type}
                            )
                        break  # successful upload
                    except Exception as e:
                        logger.warning(f"Upload failed for {relative_path} (attempt {attempt}): {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(DELAY_BETWEEN_RETRIES)
                        else:
                            logger.error(f"Final failure uploading {relative_path}")
        logger.info("✅ Zarr dataset uploaded to Cloudflare R2 successfully.")
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

def load_zarr_from_R2(bucket: str, path: str):
    """
    Downloads the specified zarr file from the provided Cloudflare R2 bucket.

    Parameters:
    - bucket: R2 bucket name in string format
    - path: path within R2 storage bucket to the desired GeoTIFF in str format

    Returns:
    - xarray.Dataset 
    """
    
    account_id = os.environ["R2_TOKEN"]
    access_key = os.environ["R2_ACCESS_KEY"]
    secret_key = os.environ["R2_SECRET_KEY"]

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": f"https://{account_id}.r2.cloudflarestorage.com"}
    )

    store = fs.get_mapper(f"{bucket}/{path}")
    ds = xr.open_zarr(store, consolidated=True, decode_timedelta="CFTimedeltaCoder")
    return ds
    

def load_tiff_from_R2(bucket: str, path: str):
    """
    Downloads the specified GeoTIFF from Cloudflare R2 bucket using s3fs + rioxarray.

    Parameters:
    - bucket: R2 bucket name
    - path: path within R2 bucket to the desired GeoTIFF

    Returns:
    - xarray.DataArray or None
    """

    account_id = os.environ["R2_TOKEN"]
    access_key = os.environ["R2_ACCESS_KEY"]
    secret_key = os.environ["R2_SECRET_KEY"]

    fs = s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": f"https://{account_id}.r2.cloudflarestorage.com"}
    )

    s3_url = f"s3://{bucket}/{path}"
    da = rioxarray.open_rasterio(s3_url, masked=True, filesystem=fs)
    return da