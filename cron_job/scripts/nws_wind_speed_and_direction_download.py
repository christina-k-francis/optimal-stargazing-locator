# -*- coding: utf-8 -*-
"""
Created on Thu May 22 19:58:55 2025

@author: Chris
"""
import gc
import xarray as xr
import pandas as pd
import numpy as np
import logging
from utils.upload_download_tools import download_grib_with_retries, upload_zarr_dataset
from utils.memory_logger import log_memory_usage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def get_wind_speed_direction():
    log_memory_usage("Before importing Wind Speed")
    
    # --- WIND SPEED ---
    logger.info("Downloading Latest NWS NDFD Wind Speed data...")
    speed_ds_1to3 = download_grib_with_retries(
        "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wspd.bin",
        variable_key="si10"
    )
    speed_ds_4to7 = download_grib_with_retries(
        "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.wspd.bin",
        variable_key="si10"
    )

    log_memory_usage("Before preprocessing Wind Speed")
    # Keep only 6-hourly timesteps
    valid_times = pd.to_datetime(speed_ds_1to3.valid_time.values)
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in [0, 6, 12, 18]]
    speed_ds_1to3 = speed_ds_1to3.isel(step=matching_idxs)
    speed_ds = xr.concat([speed_ds_1to3, speed_ds_4to7], dim="step")
    speed_ds = speed_ds.sortby("valid_time")

    log_memory_usage("Before downloading Wind Direction")

    # --- WIND DIRECTION ---
    logger.info("Downloading Latest NWS NDFD Wind Direction data...")
    dir_ds_1to3 = download_grib_with_retries(
        "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.wdir.bin",
        variable_key="wdir10"
    )
    dir_ds_4to7 = download_grib_with_retries(
        "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.wdir.bin",
        variable_key="wdir10"
    )

    # 6-hourly subset for direction
    valid_times = pd.to_datetime(dir_ds_1to3.valid_time.values)
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in [0, 6, 12, 18]]
    dir_ds_1to3 = dir_ds_1to3.isel(step=matching_idxs)
    dir_ds = xr.concat([dir_ds_1to3, dir_ds_4to7], dim="step")
    dir_ds = dir_ds.sortby("valid_time")

    log_memory_usage("After preprocessing both datasets")

    # --- COMBINE SPEED + DIRECTION ---
    logger.info("Merging Wind Speed and Wind Direction datasets...")
    masked_speed = np.ma.masked_invalid(speed_ds)
    masked_dir = np.ma.masked_invalid(dir_ds)

    direction_coords = {
        'direction': (('step', 'y', 'x'), dir_ds.data),
        'U': (('step', 'y', 'x'), -masked_speed * np.sin(np.deg2rad(masked_dir))),
        'V': (('step', 'y', 'x'), -masked_speed * np.cos(np.deg2rad(masked_dir))),
    }

    wind_ds = speed_ds.assign_coords(direction_coords)
    gc.collect()
    log_memory_usage("After assigning U/V components")

    # --- UPLOAD ZARR TO SUPABASE ---
    logger.info("Saving Resultant Wind Dataset to Cloud...")
    upload_zarr_dataset(wind_ds, "processed-data/Wind_Spd_Dir_Latest.zarr")

    return wind_ds