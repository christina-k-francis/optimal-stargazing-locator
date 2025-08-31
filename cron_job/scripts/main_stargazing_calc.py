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
import pandas as pd
from skyfield.api import load, wgs84
import pytz
import logging
import warnings
from utils.gif_tools import create_stargazing_gif
from utils.memory_logger import log_memory_usage
from utils.tile_tools import generate_stargazing_tiles 
from utils.upload_download_tools import load_zarr_from_R2, load_tiff_from_R2, upload_zarr_dataset 


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

def main():
    log_memory_usage("At start of main script")
    # 1. IMPORT RELEVANT DATA
    logger.info('Importing meteorological and astronomical data...')   
    
    # 1a. import precipitation probability dataset
    precip_da = load_zarr_from_R2("maps", "processed-data/PrecipProb_Latest.zarr")['unknown']
    precip_da.load()
    log_memory_usage("After importing precip data")
    # 1b. import sky coverage dataset
    skycover_da = load_zarr_from_R2("maps", "processed-data/SkyCover_Latest.zarr")['unknown']
    skycover_da.load()
    log_memory_usage("After importing cloud cover data")
    # 1c. import High-Res Artificial Night Sky Brightness data from David Lorenz 
    lightpollution_da = load_tiff_from_R2("maps",
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
    logger.info("Uploading Moon Dataset to Cloud...")
    upload_zarr_dataset(moonlight_da, "processed-data/Moon_Dataset_Latest.zarr")
    
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
    
    # 6. Calculate the Stargazing Index/Grades Based on Absolute Conditions
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
    w_precip = 0.4
    w_cloud = 0.75
    w_LP = 1
    w_moon = 0.2
    weight_sum = w_precip + w_cloud + w_LP + w_moon

    log_memory_usage("Before calculating the Stargazing Index")

    # 6a. Evaluate variable conditions individually by assigning letter grades
    grade_legend = {
        -1: "NA",
         0: "A+",
         1: "A",
         2: "B",
         3: "C",
         4: "D",
         5: "F"
    }

    def grade_precip(p):
        if np.isnan(p): return -1
        if p > 0.5: return 5
        elif p > 0.2: return 3
        elif p > 0.05: return 2
        else: return 0

    def grade_cloud(c):
        if np.isnan(c): return -1
        if c >= 1.0: return 5
        elif c > 0.75: return 5
        elif c > 0.5: return 4
        elif c > 0.2: return 2
        else: return 0

    def grade_lightpollution(lp):
        if np.isnan(lp): return -1
        if lp >= 0.85: return 0
        elif lp >= 0.70: return 1
        elif lp >= 0.55: return 2
        elif lp >= 0.40: return 3
        elif lp >= 0.25: return 4
        else: return 5

    def grade_moon(m):
        if np.isnan(m): return -1
        if m > 0.75: return 4
        elif m > 0.5: return 3
        elif m > 0.25: return 2
        else: return 0

    # Applying the grades to each variable
    grades_precip = xr.apply_ufunc(
        np.vectorize(grade_precip), precip_da_norm,
        dask="parallelized", output_dtypes=[np.int64]
    )
    grades_cloud = xr.apply_ufunc(
        np.vectorize(grade_cloud), skycover_da_norm,
        dask="parallelized", output_dtypes=[np.int64]
    )
    grades_lp = xr.apply_ufunc(
        np.vectorize(grade_lightpollution), lightpollution_3d,
        dask="parallelized", output_dtypes=[np.int64]
    )
    grades_moon = xr.apply_ufunc(
        np.vectorize(grade_moon), moonlight_da,
        dask="parallelized", output_dtypes=[np.int64]
    )

    # 6b. Combine the variable grades by their weights
    def combine_grades(p, c, lp, m):
        vals = []
        weights = []
        # add variable value only if it's valid
        if p != -1:
            vals.append(p); weights.append(w_precip)
        if c != -1:
            vals.append(c); weights.append(w_cloud)
        if lp != -1:
            vals.append(lp); weights.append(w_LP)
        if m != -1:
            vals.append(m); weights.append(w_moon)

        if len(vals) == 0:
            return -1

        # Special override: precip=F or clouds=F dominate
        if (p == 5 and w_precip > 0) or (c == 5 and w_cloud > 0):
            return 5

        # Weighted average of grades
        weighted_avg = np.average(vals, weights=weights)
        return int(np.rint(weighted_avg))

    # 6c. Evaluating Spatiotemporal Stargazing Conditions!
    stargazing_grades = xr.apply_ufunc(
        np.vectorize(combine_grades),
        grades_precip, grades_cloud, grades_lp, grades_moon,
        dask="parallelized", output_dtypes=[np.int64]
    )

    # Attach metadata
    stargazing_grades.attrs["legend"] = grade_legend
    stargazing_grades.attrs["description"] = "Weighted absolute grading of stargazing conditions"

    # Merge into final dataset
    stargazing_ds = xr.merge([
        stargazing_grades.rename("grade_num"),
        grades_precip.rename("grade_precip"),
        grades_cloud.rename("grade_cloud"),
        grades_lp.rename("grade_lightpollution"),
        grades_moon.rename("grade_moon")
    ], combine_attrs='no_conflicts')

    stargazing_ds = stargazing_ds.chunk(target_chunks)
    for var in stargazing_ds.data_vars:
        stargazing_ds[var].encoding.clear()

    log_memory_usage("After calculating stargazing letter grades")
    
    gc.collect # garbage collector. deletes data no longer in use
    
    # 6d. Save Stargazing DS as zarr file
    logger.info("Uploading Stargazing Evaluation Dataset to Cloud...")
    upload_zarr_dataset(stargazing_ds, "processed-data/Stargazing_Dataset_Latest.zarr")

    # 6e. Save Staragazing DS as a tileset
    logger.info(('Generating Stargazing Grade Tileset'))
    generate_stargazing_tiles(stargazing_ds['grade_num'].assign_attrs((stargazing_ds.attrs | skycover_da.attrs)),
                              "stargazing_grade", "data-layer-tiles/Stargazing_Tiles", 0.01, "gnuplot2_r")
    
    # 6f. Save Stargazing DS as a GIF
    logger.info('Creating GIF of latest Stargazing Conditions Grades forecast')
    create_stargazing_gif(stargazing_ds['grade_num'],
                          'Stargazing Conditions Grades',
                          ['N/A','A+','A','B','C','D','F'])
    
# Let's execute this main function!
main()
gc.collect() # memory saving function
