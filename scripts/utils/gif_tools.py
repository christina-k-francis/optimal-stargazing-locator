"""
Here are several functions that are useful for creating GIFs to display the
latest near-term forecast of weather variables (obtained from the NWS).
"""

import numpy as np
import pandas as pd
# organization
import gc
# cloud connection
import os
import boto3
# gif creation
import io
from PIL import Image
import pytz
# plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# custom fxs
from scripts.utils.logging_tools import log_memory_usage

mountain_tz = pytz.timezone("US/Mountain")

def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    gc.collect()
    return img

def create_nws_gif(nws_ds, cmap, cbar_label, data_title, 
                   bucket_name='optimal-stargazing-locator'):
    ticks = np.arange(0,101,10)
    images = []
    
    for time_step in range(len(nws_ds.step.values)):
        plotting_data = nws_ds[time_step]
        lat2d = nws_ds.latitude
        lon2d = nws_ds.longitude
    
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        plot = ax.pcolormesh(lon2d, lat2d, plotting_data, cmap=cmap)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5) # type: ignore
        ax.add_feature(cfeature.BORDERS, linestyle=':') # type: ignore
        ax.coastlines(resolution='110m', zorder=3) # type: ignore
        gl.top_labels=False
        plot.set_clim(0,100)
        fig.colorbar(plot, ax=ax, 
                     orientation='vertical', 
                     pad=0.1,
                     label=f'{cbar_label}', 
                     extend='neither', 
                     ticks=ticks)
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"{data_title} on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        
        # create gif snapshot
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        log_memory_usage(f"After plotting frame {time_step+1}/{len(nws_ds.step.values)}")
        gc.collect()

    # Save GIF images to buffer
    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, 
                   format='GIF', 
                   save_all=True,
                   append_images=images[1:], 
                   duration=350, 
                   loop=0)
    gif_buffer.seek(0)

    # Upload GIF to R2
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

    storage_path = f"plots/{data_title}_Latest.gif"
    s3_client.upload_fileobj(
        gif_buffer,
        bucket_name,
        storage_path,
        ExtraArgs={"ContentType": "image/gif"}
    )
    
    gif_buffer.close()
    del gif_buffer
    log_memory_usage(f"After uploading {data_title} GIF to R2")
    gc.collect()

def create_nws_temp_gif(temp_ds, cmap, cbar_label, 
                        bucket_name='optimal-stargazing-locator'):
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
    images[0].save(gif_buffer, 
                   format='GIF', 
                   save_all=True,
                   append_images=images[1:], 
                   duration=350, 
                   loop=0)
    
    # Seek to the beginning so it can be read from there
    gif_buffer.seek(0)
    
    # Upload GIF to R2
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

    storage_path = f"plots/Temp_Latest.gif"
    s3_client.upload_fileobj(
        gif_buffer,
        bucket_name,
        storage_path,
        ExtraArgs={"ContentType": "image/gif"}
    )

    gif_buffer.close()
    del gif_buffer
    log_memory_usage("After uploading Temperature GIF to R2")
    gc.collect # garbage collector. deletes data no longer in use
        
def create_stargazing_gif(stargazing_da, cbar_label, cbar_tick_labels, 
                          cmap='gnuplot2_r', bucket_name='optimal-stargazing-locator'):
    # prepare ConUSA shapefile for clipping
    USA_shp = gpd.read_file('scripts/utils/geo_ref_data/cb_2018_us_nation_5m.shp')
    conusa_mask = (-124.8, 24.4, -66.8, 49.4)
    conusa = gpd.clip(USA_shp, conusa_mask)
    # turn multipolygon into a single polygon object
    mask_geometry = conusa.geometry.union_all()
    # create the clipping path
    poly_path = geometry_to_path(mask_geometry)

    images = []
    for time_step in range(len(stargazing_da.step.values)):
        stargazing_data = stargazing_da[time_step]
        lat = stargazing_data.latitude
        lon = stargazing_data.longitude
    
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
        pc = plt.pcolormesh(lon, lat, stargazing_data, cmap=cmap,
                            transform=ccrs.PlateCarree())
        pc.set_clip_path(PathPatch(poly_path, transform=ax.transData))
        plt.clim(-1,5)
        gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.5) 
        ax.add_feature(cfeature.STATES, edgecolor='gray', linewidth=0.5) 
        ax.add_feature(cfeature.BORDERS, linestyle=':') 
        ax.coastlines(resolution='110m', zorder=3) 
        ax.set_extent([-130, -66, 22, 52], crs=ccrs.PlateCarree())
        gl.top_labels=False
        cbar = plt.colorbar(ax=ax, 
                            orientation='vertical', 
                            pad=0.1,
                            label=f'{cbar_label}', 
                            extend='neither')
        cbar.set_label(label=cbar_label, size=16, weight='bold')
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.set_yticks(np.linspace(-1,5,7))
        cbar.ax.set_yticklabels(cbar_tick_labels)
        local_dt = pd.to_datetime(stargazing_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"Stargazing Grades on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        
        img = fig2img(fig)
        images.append(img)
        plt.close(fig)
        log_memory_usage(f"After plotting frame {time_step+1}/{len(stargazing_da.step.values)}")
        gc.collect()

    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, 
                   format='GIF', 
                   save_all=True,
                   append_images=images[1:], 
                   duration=350, 
                   loop=0)
    gif_buffer.seek(0)

    # Upload GIF to R2
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

    storage_path = f"plots/Stargazing_Dataset_Latest.gif"
    s3_client.upload_fileobj(
        gif_buffer,
        bucket_name,
        storage_path,
        ExtraArgs={"ContentType": "image/gif"}
    )

    gif_buffer.close()
    del gif_buffer
    log_memory_usage("After uploading Stargazing Letter Grades GIF to R2")
    gc.collect()

def create_moon_gif(moon_ds, cmap, cbar_label, data_title, 
                    bucket_name='optimal-stargazing-locator'):
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
        plt.colorbar(ax=ax, 
                     orientation='vertical', 
                     pad=0.1, 
                     ticks=ticks,
                     label=f'{cbar_label}', 
                     extend='neither')
        local_dt = pd.to_datetime(plotting_data.valid_time.values).tz_localize(mountain_tz)
        ax.set_title(f"{data_title} on {local_dt.strftime('%Y-%m-%d %H:%M MT')}")
        
        img = fig2img(fig)
        images.append(img)
        log_memory_usage(f"After plotting frame {time_step+1}/{len(moon_ds.step.values)}")
        plt.close(fig)
        gc.collect()

    gif_buffer = io.BytesIO()
    images[0].save(gif_buffer, format='GIF', save_all=True,
                   append_images=images[1:], duration=350, loop=0)
    gif_buffer.seek(0)

    # Upload GIF to R2
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

    storage_path = f"plots/Stargazing_Dataset_Latest.gif"
    s3_client.upload_fileobj(
        gif_buffer,
        bucket_name,
        storage_path,
        ExtraArgs={"ContentType": "image/gif"}
    )

    gif_buffer.close()
    del gif_buffer
    gc.collect()

def geometry_to_path(geom):
    """Convert shapely geometry to matplotlib Path"""
    if geom.geom_type == 'Polygon':
        # Single polygon
        vertices = np.array(geom.exterior.coords)
        codes = np.ones(len(vertices), dtype=Path.code_type) * Path.LINETO
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        
        # Handle holes
        for interior in geom.interiors:
            hole_verts = np.array(interior.coords)
            hole_codes = np.ones(len(hole_verts), dtype=Path.code_type) * Path.LINETO
            hole_codes[0] = Path.MOVETO
            hole_codes[-1] = Path.CLOSEPOLY
            vertices = np.vstack([vertices, hole_verts])
            codes = np.hstack([codes, hole_codes])
        
        return Path(vertices, codes)
    
    elif geom.geom_type == 'MultiPolygon':
        # Multiple polygons - combine all into one path
        vertices_list = []
        codes_list = []
        
        for polygon in geom.geoms:
            # Exterior
            verts = np.array(polygon.exterior.coords)
            codes = np.ones(len(verts), dtype=Path.code_type) * Path.LINETO
            codes[0] = Path.MOVETO
            codes[-1] = Path.CLOSEPOLY
            vertices_list.append(verts)
            codes_list.append(codes)
            
            # Holes
            for interior in polygon.interiors:
                hole_verts = np.array(interior.coords)
                hole_codes = np.ones(len(hole_verts), dtype=Path.code_type) * Path.LINETO
                hole_codes[0] = Path.MOVETO
                hole_codes[-1] = Path.CLOSEPOLY
                vertices_list.append(hole_verts)
                codes_list.append(hole_codes)
        
        vertices = np.vstack(vertices_list)
        codes = np.hstack(codes_list)
        return Path(vertices, codes)
