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
from rasterio.enums import ColorInterp
from storage3 import create_client
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import rasterio
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

def generate_tiles_from_zarr(ds, layer_name, supabase_prefix, sleep_secs, colormap_name="viridis"):
    """
    Converts a Zarr dataset to colored raster tiles per time step using a Matplotlib colormap.

    Parameters:
    - ds (xarray data array): xarray dataset with dimensions [step, y, x]
    - layer_name (str): Label for the tiles (e.g., "cloud_coverage")
    - supabase_prefix (str): Path prefix inside Supabase bucket
    - sleep_secs (int): Delay between tile uploads
    - colormap_name (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
    """

    logger.info(f"Generating tiles for {layer_name} with colormap: {colormap_name}")

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")

    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)

    bucket_name = "maps"
    MAX_RETRIES = 5
    DELAY_BETWEEN_RETRIES = 2

    for i, timestep in enumerate(ds.step.values):
        logger.info(f"Processing timestep {i+1}/{len(ds.step.values)}")
        slice_2d = ds.isel(step=i)

        with tempfile.TemporaryDirectory() as tmpdir:
            geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.tif"
            tile_output_dir = pathlib.Path(tmpdir) / "tiles"

            # Extract transform based on attributes and GRIB metadata
            dx = slice_2d.attrs["GRIB_DxInMetres"]
            dy = slice_2d.attrs["GRIB_DyInMetres"]
            minx = -2764474.3507319926
            maxy = 3232111.7107923944

            transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)

            # Define the correct PROJ string for NDFD using CONUS LCC grid
            ndfd_proj4 = (
                "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
                "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
            )

            # Assign transform and appropriate CRS
            slice_2d.rio.write_transform(transform, inplace=True)
            slice_2d.rio.write_crs(ndfd_proj4, inplace=True)

            if "y" in slice_2d.dims:
                slice_2d = slice_2d.sortby("y", ascending=False)

            # Drop 2D geographic coordinates to prevent reproject conflict
            slice_2d = slice_2d.drop_vars(["latitude", "longitude"], errors="ignore")
            # Reproject into Web Mercator
            slice_2d = slice_2d.rio.reproject("EPSG:3857")

            # Apply colormap and save as RGB GeoTIFF
            data = slice_2d.values
            norm = Normalize(vmin=float(np.nanmin(data)), vmax=float(np.nanmax(data)))
            cmap = plt.colormaps[colormap_name]
            rgba_img = (cmap(norm(data)) * 255).astype("uint8")
            rgb_img = rgba_img[:, :, :3]  # Drop alpha

            with rasterio.open(
                geo_path,
                "w",
                driver="GTiff",
                height=rgb_img.shape[0],
                width=rgb_img.shape[1],
                count=3,
                dtype=rgb_img.dtype,
                crs="EPSG:3857",
                transform=slice_2d.rio.transform()
            ) as dst:
                for band in range(3):
                    dst.write(rgb_img[:, :, band], band + 1)

            # Generate tiles from RGB GeoTIFF
            subprocess.run([
                "gdal2tiles.py", "-z", "0-8", str(geo_path), str(tile_output_dir)
            ], check=True)

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
                            time.sleep(sleep_secs) # delay btw tile uploads to avoid cloud overload
                            break # upload successful
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

def rescale_timedelta_coords(ds, coord_name='time'):
    """
    Detects if the coordinate values are in nanoseconds and rescales them to
    larger units avoid overflow.
    Args:
        ds (xarray.Dataset or xarray.DataArray): Input data.
        coord_name (str): Coordinate to check (default='time').
    Returns:
        xarray.Dataset or xarray.DataArray: Updated object with rescaled coordinate.
    """
    if coord_name not in ds.coords:
        print(f"Coordinate '{coord_name}' not found.")
        return ds

    coord = ds.coords[coord_name]

    # Check if the values are excessively large (common for nanosecond overflow)
    if np.abs(coord.values.astype('datetime64[ns]').astype(np.int64)) > 1e18:
        one_hour = np.timedelta64(1, 'h')
        coord.values = np.timedelta64(int(ds['step'].values/one_hour))
        
        unit_start = str(coord.values.dtype).find('[')+1
        unit_end = str(coord.values.dtype).find(']')
        unit = str(coord.values.dtype)[unit_start:unit_end]
        
        if unit == 's':
            coord.encoding['units'] = "seconds"
            return coord
        elif unit == 'm':
            coord.encoding['units'] = "minutes"
            return coord
        elif unit == 'h':
            coord.encoding['units'] = "hours"
            return coord
    else:
        return coord  # No need to adjustdef rescale_timedelta_coords(ds, coord_name='time'):
    """
    Detects if the coordinate values are in nanoseconds and rescales them to
    larger units avoid overflow.
    Args:
        ds (xarray.Dataset or xarray.DataArray): Input data.
        coord_name (str): Coordinate to check (default='time').
    Returns:
        xarray.Dataset or xarray.DataArray: Updated object with rescaled coordinate.
    """
    if coord_name not in ds.coords:
        print(f"Coordinate '{coord_name}' not found.")
        return ds

    coord = ds.coords[coord_name]

    # Check if the values are excessively large (common for nanosecond overflow)
    if np.abs(coord.values.astype('datetime64[ns]').astype(np.int64)) > 1e18:
        one_hour = np.timedelta64(1, 'h')
        coord.values = np.timedelta64(int(ds['step'].values/one_hour))
        
        unit_start = str(coord.values.dtype).find('[')+1
        unit_end = str(coord.values.dtype).find(']')
        unit = str(coord.values.dtype)[unit_start:unit_end]
        
        if unit == 's':
            coord.encoding['units'] = "seconds"
            return coord
        elif unit == 'm':
            coord.encoding['units'] = "minutes"
            return coord
        elif unit == 'h':
            coord.encoding['units'] = "hours"
            return coord
    else:
        return coord  # No need to adjust

def generate_moon_tiles(ds, layer_name, supabase_prefix, sleep_secs, colormap_name="gist_yarg"):
    """
    Converts a Zarr dataset to colored raster tiles per time step using a Matplotlib colormap.

    Parameters:
    - ds (xarray data array): moon illumination xarray dataset with dimensions [step, y, x]
    - layer_name (str): Label for the tiles (e.g., "moon illumination")
    - supabase_prefix (str): Path prefix inside Supabase bucket
    - sleep_secs (int): Delay between tile uploads
    - colormap_name (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
    """

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")

    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)

    bucket_name = "maps"
    MAX_RETRIES = 5
    DELAY_BETWEEN_RETRIES = 2
    
    def get_affine_transform_from_coords(da):
        x = da.x.values
        y = da.y.values
    
        res_x = (x[-1] - x[0]) / (len(x) - 1)
        res_y = (y[0] - y[-1]) / (len(y) - 1)  # y should decrease from top to bottom
    
        transform = affine.Affine.translation(x[0] - res_x / 2, y[0] - res_y / 2) * affine.Affine.scale(res_x, -res_y)
        return transform 

    for i, timestep in enumerate(ds.step.values):
        logger.info(f"Processing timestep {i+1}/{len(ds.step.values)}")
        slice_2d = ds.isel(step=i)

        with tempfile.TemporaryDirectory() as tmpdir:
            geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.tif"
            tile_output_dir = pathlib.Path(tmpdir) / "tiles"

            # Remedying any potential chunking errors
            slice_2d = slice_2d.compute() # high memory cost
            # Assign CRS and transform
            slice_2d.rio.write_crs("EPSG:4326", inplace=True)
            transform = get_affine_transform_from_coords(slice_2d)
            slice_2d.rio.write_transform(transform, inplace=True)
            
            logger.info(f" slice_2d transform: {slice_2d.rio.transform()}")
            logger.info(f"slice_2d crs: {slice_2d.rio.crs}")

            
            # Reproject into Web Mercator
            slice_2d = slice_2d.rio.reproject("EPSG:3857")

            # Apply colormap and save as RGB GeoTIFF
            data = slice_2d.values
            norm = Normalize(vmin=0, vmax=1)
            
            cmap = plt.colormaps[colormap_name]
            rgba_img = (cmap(norm(data)) * 255).astype("uint8")
            rgb_img = rgba_img[:, :, :3]  # Drop alpha

            with rasterio.open(
                geo_path,
                "w",
                driver="GTiff",
                height=rgb_img.shape[0],
                width=rgb_img.shape[1],
                count=3,
                dtype=rgb_img.dtype,
                crs="EPSG:3857",
                transform=slice_2d.rio.transform()
            ) as dst:
                for band in range(3):
                    dst.write(rgb_img[:, :, band], band + 1)

            # Generate tiles from RGB GeoTIFF
            subprocess.run([
                "gdal2tiles.py", "-z", "0-8", str(geo_path), str(tile_output_dir)
            ], check=True)

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
                            time.sleep(sleep_secs) # delay btw tile uploads to avoid cloud overload
                            break # upload successful
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
 
def generate_stargazing_tiles(ds, layer_name, supabase_prefix, sleep_secs, colormap_name="viridis"):
    """
    Converts a Zarr dataset to colored raster tiles per time step using a Matplotlib colormap.

    Parameters:
    - ds (xarray data array): stargazing grade number xarray dataset with dimensions [step, y, x]
    - layer_name (str): Label for the tiles (e.g., "grade number, index, etc.")
    - supabase_prefix (str): Path prefix inside Supabase bucket
    - sleep_secs (int): Delay between tile uploads
    - colormap_name (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
    """

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")

    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)

    bucket_name = "maps"
    MAX_RETRIES = 5
    DELAY_BETWEEN_RETRIES = 2

    for i, timestep in enumerate(ds.step.values):
        logger.info(f"Processing timestep {i+1}/{len(ds.step.values)}")
        slice_2d = ds.isel(step=i)

        with tempfile.TemporaryDirectory() as tmpdir:
            geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.tif"
            tile_output_dir = pathlib.Path(tmpdir) / "tiles"

            # Extract transform based on attributes and GRIB metadata
            dx = slice_2d.attrs["GRIB_DxInMetres"]
            dy = slice_2d.attrs["GRIB_DyInMetres"]
            minx = -2764474.3507319926
            maxy = 3232111.7107923944

            transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)

            # Define the correct PROJ string for NDFD using CONUS LCC grid
            ndfd_proj4 = (
                "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
                "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
            )

            # Assign transform and appropriate CRS
            slice_2d.rio.write_transform(transform, inplace=True)
            slice_2d.rio.write_crs(ndfd_proj4, inplace=True)

            if "y" in slice_2d.dims:
                slice_2d = slice_2d.sortby("y", ascending=False)

            # Modifying the step coord to avoid overflow error
            slice_2d['step'] = rescale_timedelta_coords(slice_2d, "step")

            # Drop 2D geographic coordinates to prevent reproject conflict
            slice_2d = slice_2d.drop_vars(["latitude", "longitude"], errors="ignore")
            # Remedying any potential chunking errors
            slice_2d = slice_2d.compute() # high memory cost
            # Reproject into Web Mercator
            slice_2d = slice_2d.rio.reproject("EPSG:3857")

            # Apply colormap and save as RGB GeoTIFF
            data = slice_2d.values
            norm = Normalize(vmin=-1, vmax=5)
            
            cmap = plt.colormaps[colormap_name]
            rgba_img = (cmap(norm(data)) * 255).astype("uint8")
            rgb_img = rgba_img[:, :, :3]  # Drop alpha

            with rasterio.open(
                geo_path,
                "w",
                driver="GTiff",
                height=rgb_img.shape[0],
                width=rgb_img.shape[1],
                count=3,
                dtype=rgb_img.dtype,
                crs="EPSG:3857",
                transform=slice_2d.rio.transform()
            ) as dst:
                for band in range(3):
                    dst.write(rgb_img[:, :, band], band + 1)
                dst.colorinterp = (
                    ColorInterp.red,
                    ColorInterp.green,
                    ColorInterp.blue

                )

            # Generate tiles from RGB GeoTIFF
            subprocess.run([
                "gdal2tiles.py", "-z", "0-8", str(geo_path), str(tile_output_dir)
            ], check=True)

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
                            time.sleep(sleep_secs) # delay btw tile uploads to avoid cloud overload
                            break # upload successful
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
 