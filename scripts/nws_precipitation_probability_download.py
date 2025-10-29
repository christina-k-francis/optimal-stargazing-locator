# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:51:03 2025

@author: Chris
"""
###
"""
    This script downloads Precipitation Probability data from the NWS NDFD. 
    The agency updates sky coverage forecast data over the continental 
    United States every 6 hours. Here, it is prepocessed and converted to xarray
    format for manipulation.
    
"""
###

import gc
import xarray as xr
import pandas as pd
import numpy as np
from scripts.utils.logging_tools import logging_setup
from scripts.utils.upload_download_tools import download_grib_with_retries, upload_zarr_dataset

def get_precip_probability():
    logger = logging_setup()
    # URLs from NOAA NWS NDFD and grib cloud paths
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.pop12.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.pop12.bin" 
    
    logger.info("Downloading Latest Precipitation Probability data...")
    
    ds_1thru3 = download_grib_with_retries(url_days_1thru3,'unknown')
    ds_4thru7 = download_grib_with_retries(url_days_4thru7,'unknown')

    if ds_1thru3 is None or ds_4thru7 is None:
        logger.error("One or both downloads failed. Aborting precipitation processing.")
        return None

    logger.info("Preprocessing data...")
    
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 12-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
       
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("step")

    #Expand the step dimension, so 12-hourly data -> 6-hourly
    expanded_data = []
    expanded_times = []
    
    for step in range(len(combined_ds["valid_time"])):
        # current precip + time values
        p_val = combined_ds.isel(step=step).drop_vars("step") # remove inherited step
        time1 = combined_ds['valid_time'].values[step]
        # create 6 hour time steps
        time2 = time1 + np.timedelta64(6, 'h')
        
        # Split into two 6-hour values (uniform assumption)
        expanded_data.extend([p_val/2,p_val/2])
        expanded_times.extend([time1, time2])
    # concat into a single dataset
    expanded_precip = xr.concat(expanded_data, dim='step')

    # calculate the step dim as timedelta object from the first valid_time entry
    reference_time = np.datetime64(expanded_times[0])
    step_timedeltas = [np.timedelta64(t - reference_time) for t in expanded_times]

    # assign new step and valid_time coordinates to ensure no duplicates
    expanded_precip = expanded_precip.assign_coords({
        "valid_time": ("step", expanded_times),
        "step": step_timedeltas  # timedelta64 values relative to reference time
        })
    # Preserving valuable attribute info
    expanded_precip.attrs.update(combined_ds.attrs)
    
    logger.info(f"steps: {len(expanded_precip.step)}, y: {len(expanded_precip.y)}, x: {len(expanded_precip.x)}")
    
    gc.collect()
    
    # Uploading zarr file to storage bucket
    upload_zarr_dataset(expanded_precip, "processed-data/PrecipProb_Latest.zarr")

    return expanded_precip