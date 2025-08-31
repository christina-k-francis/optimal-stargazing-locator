# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:17:00 2025

@author: Chris
"""
###
"""
    This script contains functions for downloading Temperature (kelvin) data 
    from the NWS NDFD. The agency updates temperature forecast data over 
    the continental United States every 6 hours. Here, it is prepocessed and 
    converted to xarray format for manipulation.
    
"""
###

import gc
import xarray as xr
import pandas as pd
import logging
import warnings
from utils.upload_download_tools import download_grib_with_retries, upload_zarr_dataset

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

def get_temperature():
    # URLs from NOAA NWS NDFD for grib files
    url_days_1thru3 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
    url_days_4thru7 = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.004-007/ds.temp.bin"
   
    logger.info("Downloading Latest NWS NDFD Temperature data...")
    
    ds_1thru3 = download_grib_with_retries(url_days_1thru3,'t2m')
    ds_4thru7 = download_grib_with_retries(url_days_4thru7,'t2m')

    if ds_1thru3 is None or ds_4thru7 is None:
        logger.error("One or both downloads failed. Aborting temperature processing.")
        return None
    
    logger.info("Preprocessing data...")
    
    # Getting 6-hourly data
    valid_times = pd.to_datetime(ds_1thru3.valid_time.values)
    # Identify indices where hour is @ 6-hr intervals and use it for subsetting
    desired_hours = [0, 6, 12, 18]
    matching_idxs = [i for i, t in enumerate(valid_times) if t.hour in desired_hours]
    ds_1thru3_6hr = ds_1thru3.isel(step=matching_idxs)
    
    # Merging datasets
    combined_ds = xr.concat([ds_1thru3_6hr, ds_4thru7], dim="step")
    # sorting data in sequential order
    combined_ds = combined_ds.sortby("step")
    
    # Adding Celsius and Farenheit values as coordinates
    combined_ds = combined_ds.assign_coords({"fahrenheit":(["step", "y","x"],
                                                           (((combined_ds.values-273.15)*1.8)+32))})
    combined_ds = combined_ds.assign_coords({"celsius":(["step", "y","x"],
                                                        (combined_ds.values-273.15))})
    
    logger.info(f"steps: {len(combined_ds.step)}, y: {len(combined_ds.y)}, x: {len(combined_ds.x)}")
    
    gc.collect()

    logger.info("Saving Resultant Dataset to Cloud...")
    upload_zarr_dataset(combined_ds, "processed-data/Temp_Latest.zarr")

    return combined_ds