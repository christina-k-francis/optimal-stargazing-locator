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
import gc
import rioxarray
import pandas as pd
from skyfield.api import load, wgs84
import pytz
import logging
import warnings
from utils.gif_tools import create_nws_gif
from utils.memory_logger import log_memory_usage
from utils.tile_tools import generate_stargazing_tiles, generate_tiles_from_zarr
from utils.upload_download_tools import load_zarr_from_supabase,load_tiff_from_supabase,upload_zarr_dataset 


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

def main():
    log_memory_usage("At start of main script")
    # 1. IMPORT RELEVANT DATA
    logger.info('Importing meteorological and astronomical data...')   
    
    # 1a. import precipitation probability dataset
    precip_da = load_zarr_from_supabase("maps", "processed-data/PrecipProb_Latest.zarr")['unknown']
    precip_da.load()
    log_memory_usage("After importing precip data")
    # 1b. import sky coverage dataset
    skycover_da = load_zarr_from_supabase("maps", "processed-data/SkyCover_Latest.zarr")['unknown']
    skycover_da.load()
    log_memory_usage("After importing cloud cover data")
    # 1c. import High-Res Artificial Night Sky Brightness data from David Lorenz 
    lightpollution_da = load_tiff_from_supabase("maps",
                        "light-pollution-data/zenith_brightness_v22_2024_ConUSA.tif")
    lightpollution_da.load()
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

    gc.collect # garbage collector. deletes data no longer in use
    logger.info("Normalizing + Preprocessing Datasets...")
    
    # 2. NORMALIZE SKY COVERAGE DATA ON 0-1 SCALE
    # Normalize the sky coverage data on a scale (0=blue skies, 1=totally cloudy)
    # Ultimately, this negatively contributes to stargazing conditions
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
    precip_da_norm = precip_da / 100.0
    
    # 3a Convert longitudes from 0–360 to -180–180
    precip_da_norm = precip_da_norm.assign_coords(
        longitude=((precip_da_norm.longitude + 180) % 360) - 180)
    # 3b. Making lat/lon dimensions 1D instead of 2D
    precip_da_norm = precip_da_norm.assign_coords(
            latitude=("y", precip_da_norm.latitude[:, 0].data),
            longitude=("x", precip_da_norm.longitude[0, :].data))
    log_memory_usage("After normalizing precip data")
    
    # 3d. Ensuring that NWS datasets cover same forecast datetimes
    precip_da_norm = precip_da_norm[np.isin(precip_da_norm['valid_time'].values,
                                             skycover_da_norm['valid_time'].values)]
    skycover_da_norm = skycover_da_norm[np.isin(skycover_da_norm['valid_time'].values,
                                                 precip_da_norm['valid_time'].values)]
    
    # 3e. Ensuring step coordinate values are also shared
    precip_da_norm = precip_da_norm.assign_coords(step=skycover_da_norm['step'])

    gc.collect # garbage collector. deletes data no longer in use
    
    # 4. PREPROCESSING MOON ILLUMINATION DATA 
    # 4a. Illumination fraction is already normalized from 0to1
    log_memory_usage("Before calculating Moon data")

    # Mountain Time zone - NWS data is in MT
    mountain_tz = pytz.timezone("US/Mountain")
    # Desired 6-hourly time steps
    time_steps = skycover_da_norm["valid_time"].values
    # Initialize output array
    moonlight_array = np.zeros((len(time_steps), len(coarse_lats),
                                len(coarse_lons)), dtype=np.float32)
    
    # Calculate Moon Illumination over coarse grid
    for i, datetime in enumerate(time_steps):
        # Timestamps are in Mountain Time but naive — explicitly localize as Mountain Time
        aware_dt_mt = pd.to_datetime(datetime).tz_localize(mountain_tz)
        # Convert to Skyfield compatible UTC
        aware_dt_utc = aware_dt_mt.astimezone(pytz.UTC)
        t_sf = ts.utc(aware_dt_utc)
    
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

    # 4b. Save Moon Illumination+Altitude data as zarr file
    logger.info("Uploading Moon Dataset/GIF/Tileset to Cloud...")
    upload_zarr_dataset(moonlight_da, "processed-data/Moon_Dataset_Latest.zarr")
    # 4c. Create GIF of Moon Data
    logger.info("Creating GIF of moonlight data")
    create_nws_gif(moonlight_da, "gist_yarg", "Moonlight (%)",
                    "Moon Illumination + Altitude")
    # 4d. Saving Moon Data as a Tileset
    logger.info("Generating Tileset of Moon Data")
    generate_tiles_from_zarr(moonlight_da, "moon_illumination", "data-layer-tiles/Moon_Tiles", 0.01, "gist_yarg")
    
    gc.collect # garbage collector. deletes data no longer in use

    logger.info('Preprocessing high-res light pollution data...')
    log_memory_usage("Before calculating light pollution data")

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
    log_memory_usage("After calculating/interpolating light pollution data")

    gc.collect # garbage collector. deletes data no longer in use
    
    # 6. Calculate the Stargazing Index as the Sum of Weighted Values
    logger.info("Evaluating Stargazing Conditions...")
    # Ensuring that datasets are aligned chunk-wise
    target_chunks = {
        'step': 3,
        'y': 459,  # 1377 ≈ 459 * 3
        'x': 715   # 2145 ≈ 715 * 3
    }
    
    skycover_da_norm = skycover_da_norm.chunk(target_chunks)
    precip_da_norm = precip_da_norm.chunk(target_chunks)
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
        w_precip * precip_da_norm +
        w_LP * lightpollution_3d +
        w_moon * moonlight_da
    )
    
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
                return np.nan  # NA
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
            np.nan: "NA",
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
    log_memory_usage("After converting indices to letter grades")
    
    # 6c. Merge stargazing indices and letter grades into single dataset
    stargazing_ds = xr.merge([stargazing_index.rename("index"),
                              stargazing_grades.rename("grade_num")],
                             combine_attrs='no_conflicts')
    
    # Remedying likely chunk mismatching
    stargazing_ds = stargazing_ds.chunk(target_chunks)
    for var in stargazing_ds.data_vars:
        stargazing_ds[var].encoding.clear()

    # Passing on GRIB attribute data as well
    stargazing_ds.attrs.update((stargazing_ds.attrs | skycover_da.attrs))

    gc.collect # garbage collector. deletes data no longer in use
    
    # 6d. Save Stargazing DS as zarr file
    logger.info("Uploading Stargazing Evaluation Dataset to Cloud...")
    upload_zarr_dataset(stargazing_ds, "processed-data/Stargazing_Dataset_Latest.zarr")

    # 6e. Save Staragazing DS as a tileset
    logger.info("Generating Stargazing Tileset")
    generate_stargazing_tiles(stargazing_ds['grade_num'].assign_attrs((stargazing_ds.attrs | skycover_da.attrs)), 
                              "stargazing_grade", "data-layer-tiles/Stargazing_Tiles", 0.01, "gnuplot2_r")

    # 6f. Saving a GIF of stargazing condition grades
    create_nws_gif(stargazing_ds['grade_num'], "gnuplot2_r", "Stargazing Grades",
                    "Stargazing Conditions Evaluation Grades")


    
# Let's execute this main function!
main()
gc.collect() # memory saving function
