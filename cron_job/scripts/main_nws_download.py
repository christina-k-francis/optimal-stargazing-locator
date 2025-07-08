# main_download_nws.py
"""
Created on Fri May 16 12:38:25 2025

@author: Chris
"""
###
"""
    This is script needs to be run every 6 hours to retrieve the latest
    meteorological data from the U.S. National Weather Service. 
    .GIFs of the forecasts are created in this process.
"""
###
import affine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypalettes import load_cmap
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pytz
import io
import os
import gc
import time
import subprocess
import logging
import warnings
import psutil
import pathlib
import tempfile
from storage3 import create_client

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

# Functions for retrieving + preprocessing the latest data
from .nws_sky_coverage_download import get_sky_coverage
from .nws_precipitation_probability_download import get_precip_probability
from .nws_relative_humidity_download import get_relhum_percent
from .nws_average_temperature_download import get_temperature
from .nws_wind_speed_and_direction_download import get_wind_speed_direction

# All NWS data is in Mountain Time
mountain_tz = pytz.timezone("US/Mountain")

# Helpful functions
def log_memory_usage(stage: str):
    """Logs the RAM usage (RSS Memory) at it's position in the script"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"[MEMORY] RSS memory usage {stage}: {mem:.2f} MB ")

def fig2img(fig):
    """Converts a Matplotlib figure to a Pillow image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    gc.collect() # garbage collector. deletes objects that are no longer in use
    return img

def create_nws_gif(nws_ds, cmap, cbar_label, data_title):
    """ Creates a GIF of all time steps in NWS dataset """
    ticks = np.arange(0,101,10)
    images = []
    # Create plot/GIF frame
    for time_step in range(len(nws_ds.step.values)):
        plotting_data = nws_ds[time_step]
        lat2d = nws_ds.latitude
        lon2d = nws_ds.longitude
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1, projection= ccrs.PlateCarree()) 
        plt.pcolormesh(lon2d, lat2d, plotting_data, cmap=cmap)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5)  
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.coastlines(resolution='110m', zorder=3) 
        gl.top_labels=False 
        plt.clim([0,100]) 
        plt.colorbar(ax=ax, orientation='vertical', pad=0.1,
                     label=f'{cbar_label}', extend='neither', ticks=ticks) 
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"{data_title} on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        logger.info(f'{time_step+1}/{len(nws_ds.step.values)} GIF frames plotted')
        log_memory_usage(f"After plotting timestep {time_step+1}")
        gc.collect() # garbage collected at end of gif frame
    # Create GIF of plots/frames using Pillow
    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    # Seek to the beginning so it can be read from there
    gif_buffer.seek(0)
    
    # Cloud Access
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    bucket_name = "maps"
    storage_path_prefix = f"plots/{data_title}_Latest.gif"
       
    # Initialize SupaBase Storage Connection
    storage = create_client(f"{database_url}/storage/v1", 
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)
    
    log_memory_usage("Before uplpading GIF to supabase")
    
    # Upload buffer contents to cloud
    storage.from_(bucket_name).upload( 
        storage_path_prefix, 
        gif_buffer.read(), 
        {"content-type": "image/gif",
         "x-upsert":"true"})
    
    log_memory_usage("After uploading GIF to supabase")
    
    gif_buffer.close()
    del gif_buffer
    logger.info(f'GIF of Latest {data_title} forecast saved to Cloud')
    gc.collect() # cleaning up files that are no longer useful


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
                "gdal2tiles.py", "-z", "0-8", str(vrt_path), str(tile_output_dir)
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
                            time.sleep(sleep_secs)  # 50ms = 0.05sec pause between tile uploads
                            break  # Upload successful
                        except Exception as e:
                            logger.error(f"Upload failed (attempt {attempt}): {e}")
                            if attempt < MAX_RETRIES:
                                time.sleep(DELAY_BETWEEN_RETRIES)
                            else:
                                raise e
            logger.info(f"Tiles for timestep {timestamp_str} uploaded to Supabase")
            del slice_2d
            gc.collect() # deleting data that's no longer needed
    gc.collect() # Garbage Collector!
                
def main_download_nws():
    log_memory_usage("At the Start of main_download_nws")
    
    # 1. Retrieving and Preprocessing latest Sky Coverage data
    log_memory_usage("Before importing Sky Cover data")
    skycover_ds = get_sky_coverage()
    log_memory_usage("After importing Sky Cover data")
    gc.collect # garbage collector. deletes data no longer in use
    # Creating Forecast GIF
    create_nws_gif(skycover_ds, load_cmap("Bmsurface"), 
                   "Percentage of Sky Covered by Clouds", 
                   "Cloud Coverage")
    log_memory_usage("After creating GIF")
    gc.collect # garbage collector. deletes data no longer in use
    # Saving each timestep as a map tile
    generate_tiles_from_zarr(
    ds=skycover_ds,
    layer_name="cloud_coverage",
    supabase_prefix="data-layer-tiles/SkyCover_Tiles",
    sleep_secs=0.05) # 50 ms
    log_memory_usage("After creating tiles for Sky Cover timesteps")
    del skycover_ds
    gc.collect() # garbage collector. deletes objects that are no longer in use
    log_memory_usage("After DEL ds + Cleanup")
    
    # 2. Retrieving and Preprocessing latest Precipitation data
    log_memory_usage("Before importing Precip. dataset")
    precip_ds = get_precip_probability()
    log_memory_usage("After importing Precip. Dataset")
    gc.collect # garbage collector. deletes data no longer in use
    # Creating Forecast GIF
    create_nws_gif(precip_ds, load_cmap("LightBluetoDarkBlue_7"),
        "Precipitation Probability (%)",
        "Precipitation Probability")
    log_memory_usage("After creating GIF")
    gc.collect # garbage collector. deletes data no longer in use
    # Saving each timestep as a map tile
    generate_tiles_from_zarr(
    ds=precip_ds,
    layer_name="precip_probability",
    supabase_prefix="data-layer-tiles/PrecipProb_Tiles",
    sleep_secs=0.035)
    log_memory_usage("After creating tiles for Precip timesteps")
    del precip_ds
    gc.collect() # garbage collector. deletes objects that are no longer in use
    log_memory_usage("After DEL ds + Garbage Collector Cleanup")
    
    # 3. Retrieving and Preprocessing latest Relative Humidity data
    log_memory_usage("Before importing Rel. Humidity dataset")
    rhum_ds = get_relhum_percent()
    log_memory_usage("After importing Rel. Humidity dataset")
    gc.collect # garbage collector. deletes data no longer in use
    # Creating Forecast GIF
    create_nws_gif(rhum_ds, "pink_r",
        "Relative Humidity (%)",
        "Relative Humidity")
    log_memory_usage("After creating GIF")
    gc.collect # garbage collector. deletes data no longer in use
    # Saving each timestep as a map tile
    generate_tiles_from_zarr(
    ds=rhum_ds,
    layer_name="rel_humidity",
    supabase_prefix="data-layer-tiles/RelHumidity_Tiles",
    sleep_secs=0.05)
    log_memory_usage("After creating tiles for RelHum timesteps")
    del rhum_ds
    gc.collect() # garbage collector. deletes objects that are no longer in use
    log_memory_usage("After DEL ds + Garbage Collector Cleanup")
    
    # 4. Retrieving and Preprocessing latest Temperature data
    log_memory_usage("Before importing Temp. dataset")
    temp_ds = get_temperature()
    log_memory_usage("After importing Temp. dataset")
    gc.collect # garbage collector. deletes data no longer in use
    # Creating Forecast GIF - requires custom code
    images = []
    cmap = "RdYlBu_r"
    data_title = "Temperature"
    for time_step in range(len(temp_ds.step.values)):
        # unique plotting variables
        plotting_data = temp_ds['fahrenheit'][time_step]
        lat2d = temp_ds.latitude
        lon2d = temp_ds.longitude
        # celsius equivalents to fahrenheit
        ticks_f = np.arange(0,101,10)
        ticks_c = (ticks_f - 32) * 5 / 9
    
        fig, ax = plt.subplots(figsize=(12, 6), 
                               subplot_kw={'projection': ccrs.PlateCarree()}) 
        plot = ax.pcolormesh(lon2d, lat2d, plotting_data, cmap=cmap,
                             vmin=0, vmax=100)
        # map features
        gl = ax.gridlines(draw_labels=True, x_inline=False,
                          y_inline=False, linewidth=0.5)  
        gl.top_labels=False 
        ax.add_feature(cfeature.STATES,
                       edgecolor='gray', linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.coastlines(resolution='110m', zorder=3) 
        # Defining dual-unit colorbar
        cbar = plt.colorbar(plot, ax=ax, orientation='vertical',
                            pad=0.125, format='%.0f°F', ticks=ticks_f)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.tick_params(labelsize=8)
        # Adding Celsius labels next to the Fahrenheit ticks
        vmin, vmax = ticks_f[0], ticks_f[-1]
        for f_tick, c_tick in zip(ticks_f, ticks_c):
            y_pos = (f_tick - vmin) / (vmax - vmin)
            cbar.ax.text(1.2, y_pos, f"{c_tick:.0f}°C", va='center',
                         ha='left', transform=cbar.ax.transAxes, fontsize=8)
        fig.subplots_adjust(right=0.85)  # Leave room on the right for the colorbar
        cbar.set_label("Temperature in Fahrenheit and Celsius", fontsize=9, labelpad=-45)
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"Temperature on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        plt.tight_layout()
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        logger.info(f'{time_step+1}/{len(temp_ds.step.values)} GIF frames plotted')
        log_memory_usage(f"After plotting timestep {time_step+1}")
        gc.collect() # garbage collected at end of gif frame  
    # Create GIF of plots/frames using Pillow
    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    # Seek to the beginning so it can be read from there
    gif_buffer.seek(0)
    # Cloud Access
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    bucket_name = "maps"
    storage_path_prefix = "plots/Temp_Latest.gif"
    # Initialize SupaBase Bucket Connection
    storage = create_client(f"{database_url}/storage/v1",
                            {"Authorization": f"Bearer {api_key}"},
                            is_async=False)
    log_memory_usage("Before uploading GIF to supabase")
    # Upload buffer contents to cloud
    storage.from_(bucket_name).upload(
        storage_path_prefix, 
        gif_buffer.read(), 
        {"content-type": "image/gif",
         "x-upsert":"true"})
    log_memory_usage("After uploading GIF to supabase")
    gif_buffer.close()
    del gif_buffer
    logger.info(f'GIF of Latest {data_title} forecast saved to Cloud')
    log_memory_usage("After creating GIF")
    gc.collect # garbage collector. deletes data no longer in use
    # Saving each timestep as a map tile
    generate_tiles_from_zarr(
    ds=temp_ds,
    layer_name="temperature",
    supabase_prefix="data-layer-tiles/Temp_Tiles",
    sleep_secs=0.06)
    log_memory_usage("After creating tiles for Temp timesteps")
    del temp_ds
    gc.collect() # Garbage Collector
    log_memory_usage("After DEL ds + GC Cleanup")
    
    # 5. Retrieving and Preprocessing latest Wind data
    log_memory_usage("Before importing Wind datasets")
    wind_ds = get_wind_speed_direction()
    log_memory_usage("After importing Wind datasets")
    # Map tile generation not needed (yet)
    # GIF generation not needed (yet)
    del wind_ds
    gc.collect() # RAM Saving Garbage Collector
    log_memory_usage("End of main_download_nws")
    
# execute the main script:
main_download_nws()
gc.collect() # memory saving function