"""
A script for organizing overarching prefect flows
"""


import xarray as xr
import numpy as np
import pandas as pd
from prefect import flow, task
import pytz
from skyfield.api import load, wgs84
# custom fxs
from scripts.utils.grade_tools import grade_dataset
from scripts.utils.gif_tools import create_nws_gif, create_stargazing_gif
from scripts.utils.tile_tools import generate_tiles_from_zarr, generate_stargazing_tiles
from scripts.utils.upload_download_tools import upload_zarr_dataset, load_zarr_from_R2, load_tiff_from_R2
from scripts.utils.logging_tools import logging_setup


# ----- TASKS ----------------------------------------------------------------#
@task(log_prints=True, retries=3)
def download_sky_task():
    """ download latest cloud coverage forecast from NWS  """
    from scripts.nws_sky_coverage_download import get_sky_coverage
    ds = get_sky_coverage()
    return ds

@task(log_prints=True, retries=3)
def download_precip_task():
    """ download latest precip forecast from NWS """
    from scripts.nws_precipitation_probability_download import get_precip_probability
    ds = get_precip_probability()
    return ds

@task(log_prints=True, retries=3)
def create_gif_task(ds, colormap, cbar_label, title):
    """ create a GIF of a NWS data array """
    create_nws_gif(ds, colormap, cbar_label, title)
    return True

@task(log_prints=True, retries=5)
def gen_tiles_task(ds, layer_name, R2_prefix, sleep_secs, cmap, skip_tiles=False):
    """ generate tiles from a NWS data array """
    if skip_tiles:
        logger = logging_setup()
        logger.warning(f"Skipping tile generation for {layer_name} (skip_tiles=True)")
        return True
    
    generate_tiles_from_zarr(ds, layer_name, R2_prefix, sleep_secs, cmap)
    return True

# stargazing grade calculation tasks
@task(retries=3)
def calc_LP_thresholds_task():
    """ calculate light pollution thresholds for normalization """
    # convert thresholds from Falchi paper from mcd/m² to mag/arcsec²
    falchi_thresholds_mcd = {
        "pristine": 0.176,
        "astro_limit_min": 0.188,
        "winter_milky_way_limit_min": 0.397,
        "summer_milky_way_limit_min": 0.619,
        "artificial_twilight_min": 1.07,
        "artificial_twilight_max": 1.96
    }
    # function for conversion
    def luminance_to_magarcsec2(L_mcd_m2):
        """Convert luminance in mcd/m² to mag/arcsec² using V-band."""
        L_cd_m2 = L_mcd_m2 / 1000
        return -2.5 * np.log10(L_cd_m2) + 12.576
    # applying the function
    falchi_thresholds_mag = {k: luminance_to_magarcsec2(v) for k, v in falchi_thresholds_mcd.items()}
    
    # prepare new thresholds (mag/arcsec^2) a.k.a. interpolation points
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
    return mag_thresholds, normalized_values

@task(retries=3)
def normalize_light_pollution_task(lp_data, mag_thresholds, norm_values):
    """ Normalize light pollution data based on evidence-based thresholds from 0-1 """
    lightpollution_da_norm = xr.apply_ufunc(
        np.interp,
        lp_data,
        input_core_dims=[[]],
        kwargs={
            'xp': mag_thresholds,
            'fp': norm_values,
            'left': 1.0,
            'right': 0.0
        },
        dask='parallelized',
        vectorize=True,
        output_dtypes=[float]
    )
    return lightpollution_da_norm

@task(retries=3)
def interpolate_light_pollution_task(lp_data, target_lat, target_lon):
    """ Interpolate light pollution data to fit provided NWS df grid """
    lightpollution_da_norm_resampled = lp_data.interp(
        latitude=target_lat,
        longitude=target_lon,
        method='nearest'
    )
    return lightpollution_da_norm_resampled

@task(retries=3)
def grade_stargazing(cloud_grades, precip_grades, lp_grades, moon_grades,
                     w_cloud, w_precip, w_lp, w_moon):
    """ calculate weighted average of letter grades to evaluate overall stargazing conditions """
    def combine_grades(p, c, lp, m):
        vals = []
        weights = []
        # add variable value only if it's valid
        if p != -1:
            vals.append(p); weights.append(w_precip)
        if c != -1:
            vals.append(c); weights.append(w_cloud)
        if lp != -1:
            vals.append(lp); weights.append(w_lp)
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

    grades = xr.apply_ufunc(
        np.vectorize(combine_grades),
        precip_grades, cloud_grades, lp_grades, moon_grades,
        dask="parallelized", output_dtypes=[np.int8])
    
    return grades.astype('int8')

# ----- SUBFLOWS ----------------------------------------------------------------# 
# preparing precipitation data for grade calculation
@flow(name='precipitation-forecast-prep-subflow', log_prints=True)
def precip_forecast_prep_subflow():
    logger = logging_setup()
    logger.info('Subflow: prepping precip data for grading')    
    precip_da = load_zarr_from_R2('optimal-stargazing-locator', "processed-data/PrecipProb_Latest.zarr")['unknown']
    # normalize the precipitation data on a scale (0=no rain, 1=100% Showers)
    precip_da_norm = precip_da / 100.0
    precip_da_norm = precip_da_norm.assign_coords(
        longitude=((precip_da_norm.longitude + 180) % 360) - 180)
    # Make lat/lon dimensions 1D instead of 2D
    precip_da_norm = precip_da_norm.assign_coords(
            latitude=("y", precip_da_norm.latitude[:, 0].data),
            longitude=("x", precip_da_norm.longitude[0, :].data))
    logger.info('Done!') 
    return precip_da_norm  

# preparing cloud cover data for grade calculation
@flow(name='cloud-cover-forecast-prep-subflow', log_prints=True)
def cloud_cover_forecast_prep_subflow():
    logger = logging_setup()
    logger.info('Subflow: prepping clouds cover data for grading')    
    clouds_da = load_zarr_from_R2('optimal-stargazing-locator', "processed-data/SkyCover_Latest.zarr")['unknown']
    # normalize the sky coverage data on a scale (0=blue skies, 1=totally cloudy)
    clouds_da_norm = clouds_da/100.0
    # 2a. Convert longitudes from 0–360 to -180–180
    clouds_da_norm = clouds_da_norm.assign_coords(
        longitude=((clouds_da_norm.longitude + 180) % 360) - 180)
    # make lat/lon dimensions 1D instead of 2D
    clouds_da_norm = clouds_da_norm.assign_coords(
            latitude=("y", clouds_da_norm.latitude[:, 0].data),
            longitude=("x", clouds_da_norm.longitude[0, :].data))
    logger.info('Done!')
    return clouds_da_norm

# preparing moon data for grade calculation
@flow(name='moon-data-prep-subflow', log_prints=True)
def moon_data_prep_subflow(timesteps, steps, target_lat, target_lon):
    """
    input:
        timesteps: desired 6-hourly timesteps from NWS data
        steps: steps dimension data from NWS df
        target_lat: latitude coordinate from NWS df
        target_lon: longitude coordinate from NWS df
    """
    logger = logging_setup()
    logger.info('Subflow: prepping moon data for grading')
    # importing ephemeris data
    # ephemeris: the calculated positions of a celestial body over time in documentation
    eph = load('de421.bsp')
    moon, earth, sun = eph['moon'], eph['earth'], eph['sun']
    # time variables
    ts = load.timescale()
    # NWS data is in Mountain Time
    mountain_tz = pytz.timezone("US/Mountain")
    # coarse grid definition
    coarse_lats = np.linspace(24, 50, 25)  # ~1 degree resolution
    coarse_lons = np.linspace(-125, -66, 30)
    # initialize output array
    moonlight_array = np.zeros((len(timesteps), len(coarse_lats),
                                len(coarse_lons)), dtype=np.float32)
    
    # calculate moon illumination over coarse grid
    for i, datetime in enumerate(timesteps):
        # timestamps are in Mountain Time but are not zone aware — explicitly localize as Mountain Time
        aware_dt_mt = pd.to_datetime(datetime).tz_localize(mountain_tz)
        # Convert to Skyfield compatible UTC
        aware_dt_utc = aware_dt_mt.astimezone(pytz.UTC)
        t_sf = ts.utc(aware_dt_utc)
    
        for lat_i, lat in enumerate(coarse_lats):
            for lon_j, lon in enumerate(coarse_lons):
                observer = wgs84.latlon(lat, lon)
                obs = earth + observer
    
                astrometric = obs.at(t_sf).observe(moon)
                # moon azimuth/altitude
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
            "valid_time": ("step", timesteps),
            "latitude": ('latitude', coarse_lats),
            "longitude": ('longitude', coarse_lons),
            "step": steps
        },
        name="moonlight"
    )
    # interpolate to match full-resolution grid
    moonlight_da = moonlight_coarse.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    )

    upload_zarr_dataset(moonlight_da, "processed-data/Moon_Dataset_Latest.zarr")
    logger.info('Done!')
    return moonlight_da
    
# preparing light pollution data for grade calculation
@flow(name='light-pollution-prep-subflow', log_prints=True)
def light_pollution_prep_subflow(bucket_name, target_lat, target_lon,
                                 step_size, steps, valid_time):
    """
    input:
        bucket_name: cloud bucket where lp geotiff resides
        target_lat: latitude coordinate from NWS da
        target_lon: longitude coordinate from NWS da
        step_size: size of steps dimension from NWS da
        steps: steps dimension data from NWS da
        valid_time: valid_time dimension data from NWS da
    """
    logger = logging_setup()
    logger.info('Subflow: prepping light pollution data for grading')
    # import High-Res Artificial Night Sky Brightness data from David Lorenz 
    lightpollution_da = load_tiff_from_R2(bucket_name,
                        "light-pollution-data/zenith_brightness_v22_2024_ConUSA.tif")
    lightpollution_da.load()
    # define coordinate system, clip to United States, assign lat and lon xr coords
    lightpollution_da.rio.write_crs("EPSG:3857", inplace=True)
    lightpollution_da = lightpollution_da.rio.clip_box(minx=-126, miny=24, maxx=-68, maxy=50)
    lightpollution_da = lightpollution_da.rename({'x': 'longitude', 'y': 'latitude'})

    # calculate light pollution thresholds for normalization
    mag_thresholds, normalized_values = calc_LP_thresholds_task()

    # normalize artifificial night sky brightness from 0 to 1
    lp_da_norm = normalize_light_pollution_task(lightpollution_da, mag_thresholds, normalized_values)

    # interpolate dataset, so its grid matches others
    lp_da_norm = lp_da_norm.squeeze()
    lp_da_norm_resamp = interpolate_light_pollution_task(lp_da_norm,
                                                    target_lat,
                                                    target_lon)
    
    # stack 2D LP array across a new "step" dimension
    lightpollution_3d = xr.concat(
        [lp_da_norm_resamp] * step_size, dim='step')
    lightpollution_3d = lightpollution_3d.assign_coords(
        step=steps,
        valid_time=valid_time)
    
    return lightpollution_3d

# ----- FLOWS ----------------------------------------------------------------#
# ----- Stargazing Grade Calculation Flow -----
@flow(name="main-stargazing-calc-flow", log_prints=True)
def main_stargazing_calc_flow(skip_stargazing_tiles=False):
    # suppress Dask task logging
    import logging
    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.WARNING)

    logger = logging_setup()
    logger.info('preparing meteorological and astronomical data...')
    clouds_da = cloud_cover_forecast_prep_subflow()
    precip_da = precip_forecast_prep_subflow()
   
    # ensuring that NWS datasets cover same forecast datetimes
    precip_da = precip_da[np.isin(precip_da['valid_time'].values,
                                             clouds_da['valid_time'].values)]
    clouds_da = clouds_da[np.isin(clouds_da['valid_time'].values,
                                                 precip_da['valid_time'].values)]
   
    # we have to recalculate the 'step' dim since we have a new forecast start datetime
    reference_time = clouds_da['valid_time'].values[0]
    # recalculate step for clouds
    clouds_steps = [np.timedelta64(t - reference_time) 
                    for t in clouds_da['valid_time'].values]
    clouds_da = clouds_da.assign_coords({'step': clouds_steps})
    # recalculate step for precip
    precip_steps = [np.timedelta64(t - reference_time) 
                    for t in precip_da['valid_time'].values]
    precip_da = precip_da.assign_coords({'step': precip_steps})
   
    moon_da = moon_data_prep_subflow(clouds_da["valid_time"].values, 
                                     clouds_da['step'].data,
                                     clouds_da.latitude, 
                                     clouds_da.longitude)
    lp_da = light_pollution_prep_subflow("optimal-stargazing-locator",
                                         clouds_da.latitude, 
                                         clouds_da.longitude,
                                         clouds_da.sizes['step'],
                                         clouds_da['step'],
                                         clouds_da['valid_time'])
    
    logger.info('evaluating variable effects on stargazing conditions...')
    # ensuring that datasets are aligned chunk-wise
    target_chunks = {
        'step': 3,
        'y': 459,  # 1377 ≈ 459 * 3
        'x': 715   # 2145 ≈ 715 * 3
    }
    
    clouds_da = clouds_da.chunk(target_chunks)
    precip_da = precip_da.chunk(target_chunks)
    lp_da = lp_da.chunk(target_chunks)
    moon_da = moon_da.chunk(target_chunks)

    # calculating letter grades for each variable
    clouds_grades = grade_dataset(clouds_da, 'clouds')
    precip_grades = grade_dataset(precip_da, 'precip')
    lp_grades = grade_dataset(lp_da, 'lp')
    moon_grades = grade_dataset(moon_da, 'moon')

    logger.info('calculating spatiotemporal stargazing grades')
    # variable weights
    w_precip = 0.4
    w_cloud = 0.75
    w_lp = 1
    w_moon = 0.2

    stargazing_grades = grade_stargazing(clouds_grades, precip_grades, 
                                         lp_grades, moon_grades,
                                         w_cloud, w_precip, w_lp, w_moon)

    grade_legend = {
        -1: "NA",
         0: "A+",
         1: "A",
         2: "B",
         3: "C",
         4: "D",
         5: "F"
    }

    # attach metadata
    stargazing_grades.attrs["legend"] = grade_legend
    stargazing_grades.attrs["description"] = "Weighted average grading of stargazing conditions"

    # merge into final dataset
    stargazing_ds = xr.merge([
        stargazing_grades.rename("grade_num"),
        precip_grades.rename("grade_precip"),
        clouds_grades.rename("grade_cloud"),
        lp_grades.rename("grade_lightpollution"),
        moon_grades.rename("grade_moon")
    ], combine_attrs='no_conflicts')

    # mitigating sources for error 
    for var in stargazing_ds.data_vars:
        # ensure proper numeric dtype
        stargazing_ds[var] = stargazing_ds[var].astype('int8')  
        # remove old encoding->new metadata infers current state
        stargazing_ds[var].encoding.clear()
    # ensuring chunk alignment
    stargazing_ds = stargazing_ds.chunk(target_chunks)


    logger.info("uploading stargazing evaluation dataset to cloud...")
    upload_zarr_dataset(stargazing_ds, "processed-data/Stargazing_Dataset_Latest.zarr")

    if skip_stargazing_tiles == False:
        logger.info(('generating stargazing grade tileset'))
        generate_stargazing_tiles(stargazing_ds['grade_num'].assign_attrs((stargazing_ds.attrs | clouds_da.attrs)),
                                "stargazing_grade", "data-layer-tiles/Stargazing_Tiles", 0.01, "gnuplot2_r")
        
    logger.info('creating GIF of latest stargazing condition grades forecast')
    create_stargazing_gif(stargazing_ds['grade_num'],
                          'Stargazing Conditions Grades',
                          ['N/A','A+','A','B','C','D','F']) 

# ----- Preprocessing Precipitation data -----
@flow(name='precipitation-forecast-download-flow', log_prints=True)
def precipitation_forecast_flow():
    logger = logging_setup()
    logger.info('Flow: Retrieving Precipitation Data')
    ds = download_precip_task()
    logger.info('downloaded latest forecast')
    gif_boolean = create_gif_task(ds, "ocean_r", "Precipitation Probability (%)", 
                                  "Precipitation Probability")
    if not gif_boolean:
        logger.error('GIF Creation Failed')
    logger.info('GIF saved to cloud')
    tile_boolean = gen_tiles_task(ds, "precip_probability", 
                                  "data-layer-tiles/PrecipProb_Tiles",
                                  0.01, "ocean_r", skip_tiles=True)
    if not tile_boolean:
        logger.error('Tileset generation failed')
    logger.info('Preprocessing Precipitation Complete!')
    
# ----- Preprocessing Cloud Cover data -----
@flow(name='cloud-cover-forecast-download-flow', log_prints=True)
def cloud_cover_forecast_flow():
    logger = logging_setup()
    logger.info('Flow: Retrieving Cloud Cover Data')
    ds = download_sky_task()
    logger.info('downloaded latest forecast')
    gif_boolean = create_gif_task(ds, "bone", "Sky Covered by Clouds (%)", 
                                  "Cloud Coverage")
    if not gif_boolean:
        logger.error('GIF Creation Failed')
    logger.info('GIF saved to cloud')
    tile_boolean = gen_tiles_task(ds, "cloud_coverage", 
                                  "data-layer-tiles/SkyCover_Tiles",
                                  0.05, "bone", skip_tiles=True)
    if not tile_boolean:
        logger.error('Tileset generation failed')
    logger.info('Preprocessing Cloud Cover Complete!')