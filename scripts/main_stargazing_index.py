# main.py
"""
Created on Sun May 18 16:20:39 2025

@author: Chris
"""
###
"""
    This script takes cloud coverage, precipitation probability, moon illumination,
    moon azimuth, and the latest (2024) Artificial Night Sky Brightness data 
    (from David J. Lorenz) to evaluate stargazing conditions across the continental U.S. 
    The results of the evaluation are ultimately expressed as letter grades.
    
"""
###
import xarray as xr
import numpy as np
import os
import psutil
import gc
import rioxarray
import pandas as pd
from skyfield.api import load, wgs84, utc
import tempfile
import httpx
import time
import ssl
import fsspec
import logging
import warnings
from mimetypes import guess_type
from supabase import create_client, Client

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()  # Logs to stdout (Render captures this)
    ]
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

# Helpful functions
def log_memory_usage(stage: str):
    """Logs the RAM usage (RSS Memory) at it's position in the script"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"[MEMORY] RSS memory usage {stage}: {mem:.2f} MB ")

def load_zarr_from_supabase(bucket, path):
    url_base = f"https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/{bucket}/{path}/"
    fs = fsspec.filesystem("http")
    ds = xr.open_zarr(fs.get_mapper(url_base), consolidated=True,
                      decode_timedelta='CFTimedeltaCoder')
    return ds


def load_tiff_from_supabase(bucket: str, path: str) -> xr.DataArray:
    file_url = f"https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/{bucket}/{path}"
    with httpx.Client() as client:
        r = client.get(file_url)
        r.raise_for_status()
    
        try:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(r.content)
                tmp_path = tmp.name  
                tmp.flush() # ensures data is written to disk
                
                da = rioxarray.open_rasterio(tmp_path, masked=True)
                return da
        except:
            logger.exception("geoTIFF download error")
        finally:
            os.remove(tmp.name) # ensure temp file is deleted
        

def safe_upload(supabase, bucket_name, supabase_path,
                local_file_path, file_type, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(local_file_path, 'rb') as f:
                supabase.storage.from_(bucket_name).upload( 
                    supabase_path,
                    f,
                    file_options={"content-type": file_type,
                                  "upsert": "true"})
            return True
        except ssl.SSLError as ssl_err:
            logger.error(f"SSL error on attempt {attempt+1}: {ssl_err}")
            time.sleep(2* (attempt+1))
        except Exception as e:
            logger.error(f"Failed to upload {local_file_path}: {e}")
            break
    return False

def main():
    log_memory_usage("At start of main script")
    # 1. IMPORT RELEVANT DATA
    logger.info('Preprocessing meteorological and astronomical data:')   
    
    # 1a. import precipitation probability dataset
    precip_da = load_zarr_from_supabase("maps", "processed-data/PrecipProb_Latest.zarr")['unknown']
    log_memory_usage("After importing precip data")
    # 1b. import sky coverage dataset
    skycover_da = load_zarr_from_supabase("maps", "processed-data/SkyCover_Latest.zarr")['unknown']
    log_memory_usage("After importing cloud cover data")
    # 1c. import High-Res Artificial Night Sky Brightness data from David Lorenz 
    lightpollution_da = load_tiff_from_supabase("maps",
                        "light-pollution-data/zenith_brightness_v22_2024_ConUSA.tif")
    # define coordinate system, clip to United States, assign lat and lon xr coords
    lightpollution_da.rio.write_crs("EPSG:3857", inplace=True)
    lightpollution_da = lightpollution_da.rio.clip_box(minx=-126, miny=24, maxx=-68, maxy=50)
    lightpollution_da = lightpollution_da.rename({'x': 'longitude', 'y': 'latitude'})
    log_memory_usage("After importing light pollution data")
    
    # 1d. Import astronomy data from skyfield
    # Load ephemeris data
    # ephemeris: the calculated positions of a celestial body over time documented in a data file
    eph = load('de421.bsp')
    log_memory_usage("After importing Ephemeris (e.g. Moon) data")
    moon, earth, sun = eph['moon'], eph['earth'], eph['sun']
    ts = load.timescale()
    # Coarse grid definition
    coarse_lats = np.linspace(24, 50, 25)  # ~1 degree resolution
    coarse_lons = np.linspace(-125, -66, 30)
    
    # 2. NORMALIZE SKY COVERAGE DATA ON 0-1 SCALE
    # Normalize the sky coverage data on a scale (0=blue skies, 1=totally cloudy)
    # Ultimately, this negatively contributes to stargazing conditions
    logger.info("Cloud Cover")
    skycover_da_norm = skycover_da / 100.0
    # 2a. Convert longitudes from 0–360 to -180–180
    skycover_da_norm = skycover_da_norm.assign_coords(
        longitude=((skycover_da_norm.longitude + 180) % 360) - 180)
    # 2b. Making lat/lon dimensions 1D instead of 2D
    skycover_da_norm = skycover_da_norm.assign_coords(
            latitude=("y", skycover_da_norm.latitude[:, 0].data),
            longitude=("x", skycover_da_norm.longitude[0, :].data))
    log_memory_usage("After normalizing cloud cover data")
    
    # 3. NORMALIZE PRECIP DATA ON 0-1 SCALE
    # Normalize the precipitation data on a scale (0=no rain, 1=100% Showers)
    # Ultimately, this negatively contributes to stargazing conditions
    logger.info("Precipitation")
    precip_da_norm = precip_da / 100.0
    
    # 3a Convert longitudes from 0–360 to -180–180
    precip_da_norm = precip_da_norm.assign_coords(
        longitude=((precip_da_norm.longitude + 180) % 360) - 180)
    # 3b. Making lat/lon dimensions 1D instead of 2D
    precip_da_norm = precip_da_norm.assign_coords(
            latitude=("y", precip_da_norm.latitude[:, 0].data),
            longitude=("x", precip_da_norm.longitude[0, :].data))
    # 3c. Expand the step dimension, so 12-hourly data -> 6-hourly
    expanded_data = []
    expanded_times = []
    # create new duplicate data
    for step in range(len(precip_da_norm["valid_time"])):
        # current precip values
        p_val = precip_da_norm.isel(step=step)
        time1 = precip_da_norm['valid_time'].values[step]
        # create 6 hour time steps
        time2 = time1 + np.timedelta64(6, 'h')
        expanded_times.append(time1)
        expanded_times.append(time2)
        # split precip evenly to assume uniform accumulation
        expanded_data.append(p_val/2)
        expanded_data.append(p_val/2)    
    expanded_precip = xr.concat(expanded_data, dim='step')
    expanded_precip['valid_time'].values = np.array(expanded_times)
    # ensure precip has only target times (from skycover) 
    target_times = skycover_da_norm["valid_time"].values
    target_indx = np.isin(expanded_precip['valid_time'].values, target_times)
    expanded_precip = expanded_precip[target_indx]
    log_memory_usage("After normalizing precip data")
    
    # 3d. Ensuring that both datasets cover the same forecast datetimes
    common_indx = np.isin(skycover_da_norm['valid_time'].values, expanded_precip['valid_time'].values)
    skycover_da_norm = skycover_da_norm[common_indx]
    # Ensuring ascending order
    skycover_da_norm = skycover_da_norm.sortby('step')
    expanded_precip = expanded_precip.sortby('step')
    # avoiding duplicate step values
    expanded_precip = expanded_precip.assign_coords(
        step=(skycover_da_norm['step']))
    
    # 4. PREPROCESSING MOON ILLUMINATION DATA 
    # 4a. Illumination fraction is already normalized from 0to1
    # Desired 6-hourly time steps
    time_steps = skycover_da_norm["valid_time"].values
    
    log_memory_usage("Before calculating Moon data")
    # Initialize output array
    logger.info("Moon Phase and Altitude")
    moonlight_array = np.zeros((len(time_steps), len(coarse_lats),
                                len(coarse_lons)), dtype=np.float32)
    
    # Calculate Moon Illumination over coarse grid
    for i, datetime in enumerate(time_steps):
        # ensure datetime is time-zone aware + skyfield compatible
        aware_dt = pd.to_datetime(datetime).replace(tzinfo=utc)
        t_sf = ts.utc(aware_dt)
    
        for lat_i, lat in enumerate(coarse_lats):
            for lon_j, lon in enumerate(coarse_lons):
                observer = wgs84.latlon(lat, lon)
                obs = earth + observer
    
                astrometric = obs.at(t_sf).observe(moon)
                # Moon Altitude
                alt, az, _ = astrometric.apparent().altaz()
                # Illumination fraction of moon
                illum = astrometric.apparent().fraction_illuminated(sun)
    
                # Assign illumination if moon is above horizon
                if alt.degrees > 0:
                    moonlight_array[i, lat_i, lon_j] = illum
                else:
                    moonlight_array[i, lat_i, lon_j] = 0.0
    
    moonlight_coarse = xr.DataArray(
        moonlight_array,
        dims=["step", "latitude", "longitude"],
        coords={
            "valid_time": ("step", time_steps),
            "latitude": ('latitude', coarse_lats),
            "longitude": ('longitude', coarse_lons),
            "step": skycover_da_norm['step'].data
        },
        name="moonlight"
    )
    # Interpolate to match full-resolution grid
    moonlight_da = moonlight_coarse.interp(
        latitude=skycover_da_norm.latitude,
        longitude=skycover_da_norm.longitude,
        method="linear"
    )
    log_memory_usage("After calculating Moon data")    

    logger.info('Preprocessing high-res light pollution data...')
    # 5. NORMALIZE ARTIFICIAL RADIANCE DATA ON 0-1 SCALE
    # 5a Convert Falchi et. al. thresholds in mcd/m² to mag/arcsec²
    falchi_thresholds_mcd = {
        "pristine": 0.176,
        "astro_limit_min": 0.188,
        "winter_milky_way_limit_min": 0.397,
        "summer_milky_way_limit_min": 0.619,
        "artificial_twilight_min": 1.07,
        "artificial_twilight_max": 1.96
    }
    # Function for conversion
    def luminance_to_magarcsec2(L_mcd_m2):
        """Convert luminance in mcd/m² to mag/arcsec² using V-band."""
        L_cd_m2 = L_mcd_m2 / 1000
        return -2.5 * np.log10(L_cd_m2) + 12.576
    # Applying the function
    falchi_thresholds_mag = {k: luminance_to_magarcsec2(v) for k, v in falchi_thresholds_mcd.items()}
    
    # 5b. Prepare new thresholds (mag/arcsec^2) a.k.a. interpolation points
    # and normalized values
    mag_thresholds = [
        22.0,  # pristine sky
        falchi_thresholds_mag["astro_limit_min"],
        falchi_thresholds_mag["winter_milky_way_limit_min"],
        falchi_thresholds_mag["summer_milky_way_limit_min"],
        falchi_thresholds_mag["artificial_twilight_min"],
        falchi_thresholds_mag["artificial_twilight_max"]]
    mag_thresholds.reverse() # flip values in place, so increasing
    normalized_values = [
        1.0,  # pristine night sky
        0.85,  # limited astronomy
        0.6,   # winter MilkyWay gone
        0.4,   # summer MilkyWay gone
        0.1,   # artificial twilight
        0.0    # fully urban
     ]
    normalized_values.reverse() # flip values in place, so increasing
    # 5c. Normalize artificial brightness values from 0 to 1
    lightpollution_da_norm = xr.apply_ufunc(
        np.interp,
        lightpollution_da,
        input_core_dims=[[]],
        kwargs={
            'xp': mag_thresholds,
            'fp': normalized_values,
            'left': 1.0,
            'right': 0.0
        },
        dask='parallelized',
        vectorize=True,
        output_dtypes=[float]
    )
    
    # 5d. Interpolate dataset, so its grid matches the other datasets
    lp_da = lightpollution_da_norm.squeeze()
    
    # Extract latitude and longitude from skycover
    target_lat = skycover_da_norm.latitude
    target_lon = skycover_da_norm.longitude
    
    # 5e. Interpolate using lat/lon + x/y via xarray method
    lightpollution_da_norm_resampled = lp_da.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="nearest"
    )
    # Stack 2D array across a new "step" dimension
    lightpollution_3d = xr.concat(
        [lightpollution_da_norm_resampled] * skycover_da_norm.sizes['step'],
        dim='step')
    lightpollution_3d = lightpollution_3d.assign_coords(
        step=skycover_da_norm['step'],
        valid_time=skycover_da_norm['valid_time'])
    log_memory_usage("After calculating Light-Pollu. data")
    
    # 6. Calculate the Stargazing Index as the Sum of Weighted Values
    logger.info("Evaluating Stargazing Conditions...")
    # Ensuring that datasets are aligned chunk-wise
    target_chunks = {
        'step': 4,
        'y': 345,    # 1377 ≈ 345 * 4
        'x': 537     # same as current x chunks
    }
    
    skycover_da_norm = skycover_da_norm.chunk(target_chunks)
    expanded_precip = expanded_precip.chunk(target_chunks)
    lightpollution_3d = lightpollution_3d.chunk(target_chunks)
    moonlight_da = moonlight_da.chunk(target_chunks)
    
    # variable weights
    w_precip = 0.5
    w_cloud = 0.5
    w_LP = 0.75
    w_moon = 0.3
    
    log_memory_usage("Before calculating the Stargazing Index")
    # 6a. Evaluating spatiotemporal stargazing conditions!
    stargazing_index = (
        w_cloud * skycover_da_norm +
        w_precip * expanded_precip +
        w_LP * lightpollution_3d +
        w_moon * moonlight_da
    )
    
    # Ensure dataset chunks are uniform
    stargazing_index = stargazing_index.chunk({
        'step': 2,
        'y': 173,
        'x': 537
    })
    # let's make valid_time a single chunk of its own
    stargazing_index['valid_time'] = stargazing_index['valid_time'].chunk({})
    if 'chunks' in stargazing_index['valid_time'].encoding:
        del stargazing_index['valid_time'].encoding['chunks']
    log_memory_usage("After calculating the Stargazing Index")
        
    logger.info('Converting to Letter Grades...')    
    # 6b. Convert Stargazing Indices to Letter Grades
    # Letter grades are stored numerically to ensure frontend compatibility
    def index_to_grade_calc(index_data):
        # Flatten and drop NaNs
        flat_values = index_data.values.flatten()
        valid_values = flat_values[~np.isnan(flat_values)]
    
        # Calculate percentile thresholds (low = good)
        p = np.percentile(valid_values, [5, 10, 20, 35, 50])
    
        def numeric_grade(value):
            if np.isnan(value):
                return -1  # NA
            elif value <= p[0]:
                return 0  # A+
            elif value <= p[1]:
                return 1  # A
            elif value <= p[2]:
                return 2  # B
            elif value <= p[3]:
                return 3  # C
            elif value <= p[4]:
                return 4  # D
            else:
                return 5  # F
    
        grades = xr.apply_ufunc(
            np.vectorize(numeric_grade),
            index_data,
            dask="parallelized",
            output_dtypes=[int]
        )
    
        grades.attrs["legend"] = {
            -1: "NA",
             0: "A+",
             1: "A",
             2: "B",
             3: "C",
             4: "D",
             5: "F"
        }
    
        grades.attrs["thresholds"] = {
            "A+": round(p[0], 3),
            "A": round(p[1], 3),
            "B": round(p[2], 3),
            "C": round(p[3], 3),
            "D": round(p[4], 3),
            "F": f">{round(p[4], 3)}"
        }
    
        return grades
    
    # Performing the conversion!
    stargazing_grades = index_to_grade_calc(stargazing_index)
    log_memory_usage("After comverting indices to letter grades")
    
    # 6c. Merge stargazing indices and letter grades into single dataset
    stargazing_ds = xr.merge([stargazing_index.rename("index"),
                              stargazing_grades.rename("grade_num")],
                             combine_attrs='no_conflicts')
    
    # 6d. Save Stargazing_Index as zarr file
    logger.info("Uploading Stargazing Evaluation Dataset to Cloud...")
    # Initialize SupaBase Bucket Connection
    database_url = "https://rndqicxdlisfpxfeoeer.supabase.co"
    api_key = os.environ['SUPABASE_KEY']
    storage_path_prefix = 'processed-data/Stargazing_Dataset_Latest.zarr'
    if not api_key:
        logger.error("Missing SUPABASE_KEY in environment variables.")
        raise EnvironmentError("SUPABASE_KEY is required but not set.")
    
    supabase: Client = create_client(database_url, api_key)
    
    log_memory_usage("Before recursively uploading Stargazing ds to Cloud")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # save ds to temporary file
            zarr_path = f"{tmpdir}/mydata.zarr"
            # save as scalable chunked cloud-optimized zarr file
            stargazing_ds.to_zarr(zarr_path, mode="w", consolidated=True)
            
            # recursively uploading zarr data
            for root, dirs, files in os.walk(zarr_path):
                for file in files:
                    local_file_path = os.path.join(root, file)
                
                    # Convert local path to relative path for Supabase
                    relative_path = os.path.relpath(local_file_path, zarr_path)
                    supabase_path = f"{storage_path_prefix}/{relative_path.replace(os.sep, '/')}"
                    
                    mime_type, _ = guess_type(file)
                    if mime_type == None:
                        mime_type = "application/octet-stream"
                    
                    uploaded = safe_upload(supabase, 
                                           "maps", 
                                           supabase_path, 
                                           local_file_path,
                                           mime_type)
                    if not uploaded:
                        logger.error(f"Final failure for {relative_path}")
            logger.info('Latest Stargazing Letter Grades Saved to Cloud!')
            return stargazing_ds
    except:
        logger.error("Saving final dataset failed")

# Let's execute this main function!
main()
gc.collect() # memory saving function
