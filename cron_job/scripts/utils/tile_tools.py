"""
Here are function(s) useful for tile generation of multidimensional xarray datasets/zarr files
"""

import os
import xarray as xr
import numpy as np
import pandas as pd
import tempfile
import affine
import pathlib
import gc
import time
import subprocess
import logging
import warnings
import json
import httpx
from storage3 import create_client
from .memory_logger import log_memory_usage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Redirect all warnings to the logger
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=UserWarning)

# silence packages with noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)  

def generate_tiles_from_zarr(ds, layer_name, supabase_prefix, sleep_secs):
    """
    Converts a Zarr dataset to raster tiles per time step and uploads to Supabase.
    
    Parameters:
    - ds (xarray data array): xarray dataset with dimensions [step, y, x]
    - layer_name (str): Label for the tiles (e.g., "cloud_coverage")
    - supabase_prefix (str): Path prefix inside Supabase bucket
    """
    logger.info(f"Generating tiles for {layer_name}...")

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    
    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)
    bucket_name = "maps"
    MAX_RETRIES = 5
    DELAY_BETWEEN_RETRIES = 2  # seconds

    for i, timestep in enumerate(ds.step.values):
        logger.info(f"Processing timestep {i+1}/{len(ds.step.values)}")
        slice_2d = ds.isel(step=i)

        with tempfile.TemporaryDirectory() as tmpdir:
            geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.tif"
            vrt_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.vrt"
            tile_output_dir = pathlib.Path(tmpdir) / "tiles"

            # Ensure longitude values are in -180 to 180 if necessary
            if np.nanmax(slice_2d.longitude.values) > 180:
                slice_2d = slice_2d.assign_coords(
                    longitude=((slice_2d.longitude + 180) % 360) - 180
                )

            # Extract transform based on attributes and known grid
            dx = slice_2d.attrs["GRIB_DxInMetres"]  # Grid spacing in meters (x)
            dy = slice_2d.attrs["GRIB_DyInMetres"]  # Grid spacing in meters (y)

            # Bounds from GRIB metadata
            minx = -2764474.3507319926
            maxy = 3232111.7107923944

            # Construct affine transform: assumes grid is regularly spaced, origin at top-left
            transform = affine.Affine(
                dx, 0, minx, 
                0, -dy, maxy
            )

            # Define the correct PROJ string for NDFD CONUS LCC grid
            ndfd_proj4 = (
                "+proj=lcc "
                "+lat_1=25 +lat_2=25 +lat_0=25 "
                "+lon_0=-95 "
                "+x_0=0 +y_0=0 "
                "+a=6371200 +b=6371200 "
                "+units=m +no_defs"
            )
            
            # Assign transform and true lcc CRS
            slice_2d.rio.write_transform(transform, inplace=True)
            slice_2d.rio.write_crs(ndfd_proj4, inplace=True)

            # Ensure the y-axis goes North to South
            if "y" in slice_2d.dims:
                slice_2d = slice_2d.sortby("y", ascending=False)

            # Reproject to Web Mercator (EPSG:3857)
            slice_2d = slice_2d.rio.reproject("EPSG:3857")
            # Export reprojected raster
            slice_2d.rio.to_raster(geo_path)
            
            # Scale to 8-bit VRT
            subprocess.run([
                "gdal_translate", "-of", "VRT", "-ot", "Byte",
                "-scale", str(geo_path), str(vrt_path)
            ], check=True)

            # Generate tiles with gdal2tiles
            subprocess.run([
                "gdal2tiles.py", "-z", "2-8", str(vrt_path), str(tile_output_dir)
            ], check=True)

            # Upload tiles to Supabase
            timestamp_str = pd.to_datetime(slice_2d.valid_time.values).strftime('%Y%m%dT%H')

            for root, _, files in os.walk(tile_output_dir):
                for file in files:
                    rel_path = pathlib.Path(root).relative_to(tile_output_dir)
                    upload_path = f"{supabase_prefix}/{timestamp_str}/{rel_path}/{file}"
                    local_path = pathlib.Path(root) / file

                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            with open(local_path, "rb") as f:
                                storage.from_(bucket_name).upload(
                                    upload_path,
                                    f.read(),
                                    {"content-type": "image/png", "x-upsert": "true"}
                                )
                            time.sleep(sleep_secs)  # Delay between tile uploads
                            break  # Upload successful
                        except httpx.HTTPStatusError as e:
                            try:
                                logger.error(f"Supabase error: {e.response.json()}")
                            except json.JSONDecodeError:
                                logger.error(f"Supabase non-JSON response: {e.response.text}")
                            if attempt < MAX_RETRIES:
                                logger.warning(f"Upload failed (attempt {attempt}/{MAX_RETRIES}): {e}")
                                time.sleep(DELAY_BETWEEN_RETRIES)
                        except Exception as e:
                            logger.error(f"Upload error (attempt {attempt}/{MAX_RETRIES}): {e}")
                            if attempt < MAX_RETRIES:
                                time.sleep(DELAY_BETWEEN_RETRIES)
                            else:
                                logger.error(f"❌ Final failure uploading {upload_path}")

            log_memory_usage(f"After tileset {i+1}/{len(ds.step.values)}")
            logger.info(f"✅ Tiles for timestep {timestamp_str} uploaded.")
            del slice_2d
            gc.collect()

    gc.collect() # garbage collector
 