"""
Here are several functions that are useful for creating GIFs to display the
latest near-term forecast of weather variables (obtained from the NWS).
"""

import xarray as xr
import io, gc
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
from storage3 import create_client
import pytz
import os
from .memory_logger import log_memory_usage

mountain_tz = pytz.timezone("US/Mountain")

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    gc.collect()
    return img

def create_nws_gif(nws_ds, cmap, cbar_label, data_title):
    ticks = np.arange(0,101,10)
    images = []
    for time_step in range(len(nws_ds.step.values)):
        plotting_data = nws_ds[time_step]
        lat2d = nws_ds.latitude
        lon2d = nws_ds.longitude
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        plt.pcolormesh(lon2d, lat2d, plotting_data, cmap=cmap)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.BORDERS, linestyle=':') # type: ignore
        ax.coastlines(resolution='110m', zorder=3) # type: ignore
        gl.top_labels=False
        plt.clim(0,100)
        plt.colorbar(ax=ax, orientation='vertical', pad=0.1,
                     label=f'{cbar_label}', extend='neither', ticks=ticks)
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"{data_title} on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        log_memory_usage(f"After plotting frame {time_step+1}/{len(nws_ds.step.values)}")
        gc.collect()

    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    gif_buffer.seek(0)

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"}, is_async=False)

    storage_path_prefix = f"plots/{data_title}_Latest.gif"
    storage.from_("maps").upload(
        storage_path_prefix,
        gif_buffer.read(),
        {"content-type": "image/gif", "x-upsert": "true"}
    )
    gif_buffer.close()
    del gif_buffer
    log_memory_usage(f"After uploading {data_title} GIF to Supabase")
    gc.collect()

def create_nws_temp_gif(temp_ds, cmap, cbar_label):
    # Creating Forecast GIF - requires custom code
    images = []
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
        ax.add_feature(cfeature.STATES, # type: ignore
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
        cbar.set_label(cbar_label, fontsize=9, labelpad=-65)
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"Temperature on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        plt.tight_layout()
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        log_memory_usage(f"After plotting frame {time_step+1}/{len(temp_ds.step.values)}")
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
    gif_buffer.close()
    del gif_buffer
    log_memory_usage("After uploading Temperature GIF to supabase")
    gc.collect # garbage collector. deletes data no longer in use
        
def create_moon_gif(moon_ds, cmap, cbar_label, data_title):
    ticks = np.arange(0,101,10)
    images = []
    for time_step in range(len(moon_ds.step.values)):
        plotting_data = moon_ds[time_step]
        lat2d = moon_ds.latitude
        lon2d = moon_ds.longitude
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        plt.pcolormesh(lon2d, lat2d, plotting_data, cmap=cmap)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.BORDERS, linestyle=':') 
        ax.coastlines(resolution='110m', zorder=3)
        gl.top_labels=False
        plt.clim(0,100)
        plt.colorbar(ax=ax, orientation='vertical', pad=0.1, ticks=ticks,
                     label=f'{cbar_label}', extend='neither')
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"{data_title} on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        gc.collect()

    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    gif_buffer.seek(0)

    api_key = os.environ['SUPABASE_KEY']
    if not api_key:
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    storage = create_client("https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1",
                            {"Authorization": f"Bearer {api_key}"}, is_async=False)

    storage_path_prefix = f"plots/{data_title}_Latest.gif"
    storage.from_("maps").upload(
        storage_path_prefix,
        gif_buffer.read(),
        {"content-type": "image/gif", "x-upsert": "true"}
    )
    gif_buffer.close()
    del gif_buffer
    gc.collect()
