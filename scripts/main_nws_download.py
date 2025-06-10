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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypalettes import load_cmap
from PIL import Image
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import io
import os
import logging
import warnings
from supabase import create_client, Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# Redirect all warnings to the logger
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=UserWarning)

# Functions for retrieving + preprocessing the latest data
from .nws_sky_coverage_download import get_sky_coverage
from .nws_precipitation_probability_download import get_precip_probability
from .nws_relative_humidity_download import get_relhum_percent
from .nws_average_temperature_download import get_temperature
from .nws_wind_speed_and_direction_download import get_wind_speed_direction

# Helpful functions
def fig2img(fig):
    """Converts a Matplotlib figure to a Pillow image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
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
        ax.set_title(f"{data_title} on {pd.to_datetime(plotting_data.valid_time.values).strftime('%Y-%m-%d %H:%M UTC')}")
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        logger.info(f'{time_step+1}/{len(nws_ds.step.values)} GIF frames plotted')
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
       
    # Initialize SupaBase Bucket Connection
    supabase: Client = create_client(database_url, api_key)

    # Upload buffer contents to cloud
    supabase.storage.from_(bucket_name).upload( 
        storage_path_prefix, gif_buffer.read(), 
        {"content-type": "image/gif",
         "x-upsert":"true"})
    gif_buffer.close()

    logger.info(f'GIF of Latest {data_title} forecast saved to Cloud')
    
def main_download_nws():
    # 1. Retrieving and Preprocessing latest Sky Coverage data
    skycover_ds = get_sky_coverage()
    # Creating Forecast GIF
    create_nws_gif(skycover_ds, load_cmap("Bmsurface"), 
                   "Percentage of Sky Covered by Clouds", 
                   "Cloud Coverage")
    
    # 2. Retrieving and Preprocessing latest Precipitation data
    precip_ds = get_precip_probability()
    # Creating Forecast GIF
    create_nws_gif(precip_ds, load_cmap("LightBluetoDarkBlue_7"),
        "Precipitation Probability (%)",
        "Precipitation Probability")
    
    
    # 3. Retrieving and Preprocessing latest Relative Humidity data
    rhum_ds = get_relhum_percent()
    # Creating Forecast GIF
    create_nws_gif(rhum_ds, "pink_r",
        "Relative Humidity (%)",
        "Relative Humidity")
    
    # 4. Retrieving and Preprocessing latest Temperature data
    temp_ds = get_temperature()
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
        cbar.set_label("Temperature in Fahrenheit and Celsius", fontsize=9, labelpad=-25)
        ax.set_title(f"Temperature on {pd.to_datetime(plotting_data.valid_time.values).strftime('%Y-%m-%d %H:%M UTC')}")
        plt.tight_layout()
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        logger.info(f'{time_step+1}/{len(temp_ds.step.values)} GIF frames plotted')
        
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
    supabase: Client = create_client(database_url, api_key)
    
    # Upload buffer contents to cloud
    supabase.storage.from_(bucket_name).upload(
        storage_path_prefix, gif_buffer.read(), 
        {"content-type": "image/gif",
         "x-upsert":"true"})
    
    logger.info(f'GIF of Latest {data_title} forecast saved to Cloud')
    
    # 5. Retrieving and Preprocessing latest Wind data
    wind_ds = get_wind_speed_direction()
    # No plots for this just yet
    
# execute the main script:
main_download_nws()
