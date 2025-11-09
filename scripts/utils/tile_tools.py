"""
Here are function(s) useful for tile generation of multidimensional xarray datasets/zarr files
"""

import xarray as xr
import numpy as np
import pandas as pd
from prefect import task
# tile generation
import affine
import subprocess
from osgeo_utils import gdal2tiles
from concurrent.futures import ThreadPoolExecutor, as_completed
# cloud connection
import os
import boto3
import time
# file handling
import tempfile
import pathlib
# organization
import gc
# raster visualization
import geopandas as gpd
import rasterio
from rasterio.enums import ColorInterp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# custom fxs
from scripts.utils.logging_tools import logging_setup, log_memory_usage

@task(log_prints=True, retries=3)
def upload_tile_to_r2(tile_output_dir, R2_prefix, timestamp_str, sleep_secs):

    logger = logging_setup()

    bucket_name = "optimal-stargazing-locator"
    MAX_RETRIES = 5
    DELAY_BETWEEN_RETRIES = 2

    account_id = os.environ["R2_ACCOUNT_ID"]
    access_key = os.environ["R2_ACCESS_KEY"]
    secret_key = os.environ["R2_SECRET_KEY"]

    endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    for root, _, files in os.walk(tile_output_dir):
        for file in files:
            rel_path = pathlib.Path(root).relative_to(tile_output_dir)
            upload_path = f"{R2_prefix}/{timestamp_str}/{rel_path}/{file}"
            # resolving common path error w/ .xml and .html files
            if '/./' in upload_path:
                upload_path = upload_path.replace('/./', "/")
            local_path = pathlib.Path(root) / file

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    with open(local_path, "rb") as f:
                        s3_client.upload_fileobj(
                            f,
                            bucket_name,
                            upload_path,
                            ExtraArgs={"ContentType": "image/png"}
                        )
                    time.sleep(sleep_secs)  # delay to avoid hammering R2
                    break  # upload successful
                except Exception as e:
                    logger.error(f"Upload error (attempt {attempt}/{MAX_RETRIES}) for {upload_path}: {e}")
                    if attempt < MAX_RETRIES:
                        time.sleep(DELAY_BETWEEN_RETRIES)
                    else:
                        logger.error(f"Final failure uploading {upload_path}")

@task(log_prints=True, retries=3)
def generate_tiles_from_xr(ds, layer_name, R2_prefix, sleep_secs, 
                           colormap_name="viridis", vmin=None, vmax=None):
    """
    description:
        Converts an xarray data array to colored raster tiles per timestep in parallel.

    input:
        - ds (xarray data array): xarray data array with dimensions [step, y, x]
        - layer_name (str): Label for the tiles (e.g., "cloud_coverage")
        - R2_prefix (str): Path prefix inside R2 bucket
        - sleep_secs (int): Delay between tile uploads
        - colormap_name (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
        - vmin (numeric): Minimum value for color normalization (None = auto from data)
        - vmax (numeric): Maximum value for color normalization (None = auto from data)
    """
    logger = logging_setup()
    logger.info(f"Generating tiles for {layer_name} with colormap: {colormap_name}")

    num_steps = ds.sizes['step']

    # Process timesteps in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all timesteps
        futures = {executor.submit(
            generate_single_timestep_tiles,
            ds, layer_name, R2_prefix, i, sleep_secs, 
            colormap_name, vmin, vmax
                ): i for i in range(num_steps)
            }
        
        # Wait for completion
        completed = 0
        for future in as_completed(futures):
            timestep_idx = futures[future]
            try:
                future.result()
                completed += 1
                logger.info(f"Progress: {completed}/{num_steps} timesteps complete")
            except Exception as e:
                logger.error(f"Timestep {timestep_idx} failed: {e}")
                raise
    
    logger.info(f"All tiles generated for {layer_name}")
    gc.collect() # clear excess memory
    return True

@task(log_prints=True, retries=3)
def generate_single_timestep_tiles(ds, layer_name, R2_prefix, timestep_idx, 
                                   sleep_secs, cmap, vmin=None, vmax=None,
                                   conus_shapefile='scripts/utils/geo_ref_data/cb_2018_us_nation_5m.shp'):
    """
    description:
        Generate tiles for a single timestep

    input:
        - ds (xarray data array): xarray dataset with dimensions [step, y, x]
        - layer_name (str): Label for the tiles (e.g., "cloud_coverage")
        - R2_prefix (str): Path prefix inside R2 bucket
        - timestep_idx (int): index position of the current timestamp
        - sleep_secs (int): Delay between tile uploads
        - cmap (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
        - vmin (numeric): Minimum value for color normalization (None = auto from data)
        - vmax (numeric): Maximum value for color normalization (None = auto from data
    """
    logger = logging_setup()

    # Extract data for this timestep
    slice_2d = ds.isel(step=timestep_idx)

    # Handle potential timedelta overflow
    slice_2d = _fix_timedelta_overflow(slice_2d)

    # Create temp directory for this timestep
    with tempfile.TemporaryDirectory() as tmpdir:
        geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{timestep_idx}.tif"
        tile_output_dir = pathlib.Path(tmpdir) / "tiles"

        # Extract transform based on attributes and GRIB metadata
        dx = slice_2d.attrs["GRIB_DxInMetres"]
        dy = slice_2d.attrs["GRIB_DyInMetres"]
        minx = -2764474.3507319926
        maxy = 3232111.7107923944
        transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)

        # Define CONUS Lambert Conformal Conic projection (from NDFD)
        ndfd_proj4 = (
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
            "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
        )

        # Assign transform and appropriate CRS
        slice_2d.rio.write_transform(transform, inplace=True)
        slice_2d.rio.write_crs(ndfd_proj4, inplace=True)

        # ensure y-axis is in the correct order
        if "y" in slice_2d.dims:
            slice_2d = slice_2d.sortby("y", ascending=False)

        # Drop 2D geographic coordinates to prevent reprojection conflict
        slice_2d = slice_2d.drop_vars(["latitude", "longitude"], errors="ignore")
        # Reproject into Web Mercator
        slice_2d = slice_2d.rio.reproject("EPSG:3857")

        # Apply CONUS mask to tile data
        USA_gdf = gpd.read_file(conus_shapefile)
        # prepare shapefile - clipping to conus
        conusa_mask = (-124.8, 24.4, -66.8, 49.4)
        conus_gdf = gpd.clip(USA_gdf, conusa_mask)
        conus_gdf = conus_gdf.to_crs("EPSG:3857")
            
        # Clip ds to CONUS boundary
        slice_2d = slice_2d.rio.clip(conus_gdf.geometry, all_touched=True)

        # Apply colormap and save as RGB GeoTIFF
        data = slice_2d.values

        # Use provided vmin/vmax or calculate from data
        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

        # Create mask 
        valid_mask = ~np.isnan(data)

        # defining RGB colormap
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = plt.colormaps[cmap]
        rgba_img = (colormap(norm(data)) * 255).astype("uint8")
        rgba_img[:, :, 3] = np.where(valid_mask, 255, 0)

        # create/write RGBA GeoTIFF
        with rasterio.open(
            geo_path,
            "w",
            driver="GTiff",
            height=rgba_img.shape[0],
            width=rgba_img.shape[1],
            count=4,
            dtype=rgba_img.dtype,
            crs="EPSG:3857",
            transform=slice_2d.rio.transform()
        ) as dst:
            for band in range(4):
                dst.write(rgba_img[:, :, band], band + 1)
            # set color interpretation in metadata
            dst.colorinterp = (ColorInterp.red,
                               ColorInterp.green,
                               ColorInterp.blue,
                               ColorInterp.alpha)
            
        
        # Generate tiles from RGBA GeoTIFF
        try:
            subprocess.run([
                "gdal2tiles.py",
                "-z", "0-8",  # Zoom levels
                str(geo_path),            
                str(tile_output_dir)      
            ], check=True, capture_output=True, text=True)
            
            logger.info(f"Tiles generated for timestep {timestep_idx+1}")
        except subprocess.CalledProcessError as e:
            logger.error(f"gdal2tiles failed for timestep {timestep_idx}: {e.stderr}")
            raise

        # Now, let's upload tiles to R2
        timestamp_str = pd.to_datetime(slice_2d.valid_time.values).strftime('%Y%m%dT%H')
        upload_tile_to_r2(tile_output_dir, R2_prefix, timestamp_str, sleep_secs)
        
        return True


def _fix_timedelta_overflow(ds, coord_name='time'):
    """
    description:
        Helper FX that detects if the coordinate values are in nanoseconds 
        and rescales them to larger units avoid overflow.
    input:
        - ds (xarray.Dataset or xarray.DataArray): Input data.
        - coord_name (str): Coordinate to check (default='time').
    output:
        xarray.Dataset or xarray.DataArray: Updated object with rescaled coordinate.
    """
    
    if coord_name not in ds.coords:
        print(f"Coordinate '{coord_name}' not found.")
        return ds
    
    coord = ds.coords[coord_name]
    
    # Check if values are excessively large (nanosecond overflow)
    try:
        coord_ns = coord.values.astype('datetime64[ns]').astype(np.int64)
        if np.abs(coord_ns).max() > 1e18:
            # Convert to hours to avoid overflow
            one_hour = np.timedelta64(1, 'h')
            hours = int(coord.values / one_hour)
            
            # Create new coordinate
            ds = ds.assign_coords({coord_name: np.timedelta64(hours, 'h')})
            ds.coords[coord_name].encoding['units'] = 'hours'
    except (TypeError, ValueError):
        # If conversion fails, leave as-is
        pass
    
    return ds
    
def generate_moon_tiles(ds, layer_name, R2_prefix, sleep_secs, colormap_name="gist_yarg"):
    """
    description:
        Converts a Zarr dataset to colored raster tiles per time step using a Matplotlib colormap.

    input:
        - ds (xarray data array): moon illumination xarray dataset with dimensions [step, y, x]
        - layer_name (str): Label for the tiles (e.g., "moon illumination")
        - R2_prefix (str): Path prefix inside R2 bucket
        - sleep_secs (int): Delay between tile uploads
        - colormap_name (str): Name of Matplotlib colormap (e.g., "Blues", "coolwarm")
    """
    logger = logging_setup()

    bucket_name = "optimal-stargazing-locator"
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
        
        # Select 2D slice
        slice_2d = ds.isel(step=i).squeeze(drop=True)  # shape: (y, x)
        # Compute if needed
        slice_2d = slice_2d.compute()

        with tempfile.TemporaryDirectory() as tmpdir:
            geo_path = pathlib.Path(tmpdir) / f"{layer_name}_t{i}.tif"
            tile_output_dir = pathlib.Path(tmpdir) / "tiles"

            # Explicitly name spatial dimensions
            slice_2d = slice_2d.rename({"x": "x", "y": "y"})
            
            # Ensure CRS + Transform are assigned
            transform = get_affine_transform_from_coords(slice_2d)
            slice_2d.rio.write_crs("EPSG:4326", inplace=True)
            slice_2d.rio.write_transform(transform, inplace=True)
        
            logger.info(f"Transform: {slice_2d.rio.transform()}")
            logger.info(f"CRS: {slice_2d.rio.crs}")
        
            # Confirm required spatial dims exist
            assert "x" in slice_2d.dims and "y" in slice_2d.dims, "Missing spatial dims"
        
            # Reproject
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

            account_id = os.environ["R2_ACCOUNT_ID"]
            access_key = os.environ["R2_ACCESS_KEY"]
            secret_key = os.environ["R2_SECRET_KEY"]

            endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

            s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )

            for root, _, files in os.walk(tile_output_dir):
                for file in files:
                    rel_path = pathlib.Path(root).relative_to(tile_output_dir)
                    upload_path = f"{R2_prefix}/{timestamp_str}/{rel_path}/{file}"
                    # resolving common path error w/ .xml and .html files
                    if '/./' in upload_path:
                        upload_path = upload_path.replace('/./', "/")
                    local_path = pathlib.Path(root) / file

                    for attempt in range(1, MAX_RETRIES + 1):
                        try:
                            with open(local_path, "rb") as f:
                                s3_client.upload_fileobj(
                                    f,
                                    bucket_name,
                                    upload_path,
                                    ExtraArgs={"ContentType": "image/png"}
                                )
                            time.sleep(sleep_secs)  # delay to avoid hammering R2
                            break  # upload successful
                        except Exception as e:
                            logger.error(f"Upload error (attempt {attempt}/{MAX_RETRIES}) for {upload_path}: {e}")
                            if attempt < MAX_RETRIES:
                                time.sleep(DELAY_BETWEEN_RETRIES)
                            else:
                                logger.error(f"❌ Final failure uploading {upload_path}")


            log_memory_usage(f"After tileset {i+1}/{len(ds.step.values)}")
            logger.info(f"✅ Tiles for timestep {timestamp_str} uploaded.")
            del slice_2d
            gc.collect()

    gc.collect() # garbage collector
 