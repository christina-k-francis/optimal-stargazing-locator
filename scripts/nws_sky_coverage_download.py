# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:24:08 2025

@author: Chris
"""
###
"""
    This script contains functions for downloading Sky Coverage (%) data 
    from the NWS NDFD. The agency updates sky coverage forecast data over 
    the continental United States every 6 hours. Here, it is prepocessed, 
    converted to xarray format for manipulation, and used to generate a 
    .gif of the 7 day forecast.
    
"""
###

import gc
import xarray as xr
import pandas as pd
from scripts.utils.logging_tools import logging_setup
from scripts.utils.upload_download_tools import download_grib_with_retries, upload_zarr_dataset

def get_sky_coverage():    
    logger = logging_setup()
    # URLs from NOAA NWS NDFD and grib cloud paths
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.sky.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.sky.bin"
    
    logger.info('Downloading latest Sky Coverage dataset from NWS NDFD...')
    
    ds_1thru3 = download_grib_with_retries(url_days_1thru3,'unknown')
    ds_4thru7 = download_grib_with_retries(url_days_4thru7,'unknown')

    if ds_1thru3 is None or ds_4thru7 is None:
        logger.error("One or both downloads failed. Aborting sky coverage processing.")
        return None

    logger.info('Preprocessing Data...')
    
    # Getting 6-hourly data across all days
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
    
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("step")

    logger.info(f"steps: {len(combined_ds.step)}, y: {len(combined_ds.y)}, x: {len(combined_ds.x)}")
    
    gc.collect()

    # Uploading zarr file to storage bucket
    upload_zarr_dataset(combined_ds, "processed-data/SkyCover_Latest.zarr")

    return combined_ds
    