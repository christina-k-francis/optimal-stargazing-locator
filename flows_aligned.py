"""
A script for organizing prefect flows, subflows, and tasks
"""

import gc
import xarray as xr
import numpy as np
import pandas as pd
import time
import pytz
import affine
import shapely
import geopandas as gpd
from prefect import flow, task
from skyfield.api import load, wgs84
from scipy.interpolate import RegularGridInterpolator
# custom fxs
from scripts.utils.grade_tools import grade_dataset
from scripts.utils.gif_tools import create_nws_gif, create_stargazing_gif
from scripts.utils.tile_tools import generate_tiles_from_xr
from scripts.utils.upload_download_tools import upload_zarr_dataset, load_zarr_from_R2, load_tiff_from_R2
from scripts.utils.logging_tools import logging_setup, log_memory_usage


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
def gen_tiles_task(ds, layer_name, R2_prefix, sleep_secs, cmap, 
                   vmin=None, vmax=None, skip_tiles=False):
    """ generate tiles from a NWS data array """
    if skip_tiles:
        logger = logging_setup()
        logger.warning(f"Skipping tile generation for {layer_name} (skip_tiles=True)")
        return True
    else:
        log_memory_usage(f"before generating {layer_name} tiles")
        generate_tiles_from_xr(ds, layer_name, R2_prefix, 
                               sleep_secs, cmap, vmin, vmax)
        gc.collect()
        log_memory_usage(f"after generating {layer_name} tiles")
        return True

# stargazing grade calculation tasks
@task(retries=3)
def calc_LP_thresholds_task():
    """ calculate light pollution thresholds for normalization """
    # convert thresholds from Falchi paper in mcd/m² to mag/arcsec²
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
        1.0,   # pristine night sky
        0.75,  # limited astronomy
        0.5,   # winter MilkyWay gone
        0.3,   # summer MilkyWay gone
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
            'left': 0.0,
            'right': 1.0
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
    result = grades.astype('int8')
    
    # clean up intermediate
    del grades
    gc.collect()
    
    return result

# ----- SUBFLOWS ----------------------------------------------------------------# 
# preparing precipitation data for grade calculation
@flow(name='precipitation-forecast-prep-subflow', log_prints=True)
def precip_forecast_prep_subflow():
    # Keep attributes during all operations
    xr.set_options(keep_attrs=True)
    
    logger = logging_setup()
    logger.info('Subflow: prepping precip data for grading')    
    
    # 1. Importing precip data
    precip_da = load_zarr_from_R2('optimal-stargazing-locator', "processed-data/PrecipProb_Latest.zarr")['unknown']
    
    # 2. Normalize the precipitation data on a scale (0=no rain, 1=100% Showers)
    precip_da_norm = precip_da / 100.0
    
    # 3a. Convert longitudes from 0–360 to -180–180
    precip_da_norm = precip_da_norm.assign_coords(
        longitude=((precip_da_norm.longitude + 180) % 360) - 180)
     # 3b. Ensure the proper CRS metadata is preserved
    if precip_da_norm.rio.crs is None:
        ndfd_proj4 = (
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
            "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
        )
        precip_da_norm.rio.write_crs(ndfd_proj4, inplace=True)
        
        # Set the spatial transform (from your tile generation code)
        dx = precip_da_norm.attrs["GRIB_DxInMetres"]
        dy = precip_da_norm.attrs["GRIB_DyInMetres"]
        minx = -2764474.3507319926
        maxy = 3232111.7107923944
        transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)
        precip_da_norm.rio.write_transform(transform, inplace=True)

    logger.info('Precip Prob Grading Prep Complete!') 
    return precip_da_norm  

# preparing cloud cover data for grade calculation
@flow(name='cloud-cover-forecast-prep-subflow', log_prints=True)
def cloud_cover_forecast_prep_subflow():
    # Keep attributes during all operations
    xr.set_options(keep_attrs=True)
    
    logger = logging_setup()
    logger.info('Subflow: prepping clouds cover data for grading')    
    
    # 1. Import cloud cover data
    clouds_da = load_zarr_from_R2('optimal-stargazing-locator', "processed-data/SkyCover_Latest.zarr")['unknown']
    
    # 2. normalize the sky coverage data on a 0-1 scale (0=blue skies, 1=totally cloudy)
    clouds_da_norm = clouds_da/100.0
    
    # 3a. Convert longitudes from 0–360 to -180–180
    clouds_da_norm = clouds_da_norm.assign_coords(
        longitude=((clouds_da_norm.longitude + 180) % 360) - 180)
    # 3b. Ensure the proper CRS metadata is preserved
    if clouds_da_norm.rio.crs is None:
        ndfd_proj4 = (
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
            "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
        )
        clouds_da_norm.rio.write_crs(ndfd_proj4, inplace=True)
        
        # Set the spatial transform (from your tile generation code)
        dx = clouds_da_norm.attrs["GRIB_DxInMetres"]
        dy = clouds_da_norm.attrs["GRIB_DyInMetres"]
        minx = -2764474.3507319926
        maxy = 3232111.7107923944
        transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)
        clouds_da_norm.rio.write_transform(transform, inplace=True)

    logger.info('Cloud Cover Grading Prep Complete!')
    
    return clouds_da_norm

# preparing moon data for grade calculation
@flow(name='moon-data-prep-subflow', log_prints=True)
def moon_data_prep_subflow(target_da):
    """
    input:
        target_da: NWS data array with target dims and coords
    """
    logger = logging_setup()
    logger.info('Subflow: prepping moon data for grading')
    
    # extract vital data from target NWS da
    timesteps = target_da['valid_time'].values

    # load ephemeris data
    # ephemeris: the calculated positions of a celestial body over time in documentation
    eph = load('de421.bsp')
    moon, earth, sun = eph['moon'], eph['earth'], eph['sun']
    
    # time variables
    ts = load.timescale()
    # NWS data is in UTC time zone
    utc_tz = pytz.timezone("UTC")
    
    # let's calculate lunar data across a coarse grid to save time
    coarse_lats = np.linspace(24, 50, 25)  # ~1 degree resolution
    coarse_lons = np.linspace(-126, -66, 30)
    
    # initialize output array
    moonlight_array = np.zeros((len(timesteps), len(coarse_lats),
                                len(coarse_lons)), dtype=np.float32)

    logger.info('Calculating moon illumination over an approx. 1°x1° coarse grid...')
    # calculate moon illumination across timesteps over coarse grid
    for i, datetime in enumerate(timesteps):
        # Localize timestamp to UTC (skyfield compatibility)
        aware_dt_utc = pd.to_datetime(datetime).tz_localize(utc_tz)
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

    # get target 2D coordinates from NWS da
    lats_2d = target_da.latitude.values  # Shape: (y, x)
    lons_2d = target_da.longitude.values  # Shape: (y, x)

    logger.info('Interpolating lunar data to high-resolution NWS grid')
    moonlight_highres = np.zeros((len(timesteps), lats_2d.shape[0], lats_2d.shape[1]), 
                                  dtype=np.float32)
    
    # interpolating each timestep
    for i in range(len(timesteps)):
        # Create interpolator for this timestep
        interpolator = RegularGridInterpolator(
            (coarse_lats, coarse_lons),
            moonlight_array[i],
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )

        # Create array of (lat, lon) points to interpolate
        points = np.column_stack([lats_2d.ravel(), lons_2d.ravel()])
        # Interpolate
        interpolated_flat = interpolator(points)
        # Reshape back to 2D grid
        moonlight_highres[i] = interpolated_flat.reshape(lats_2d.shape)
    # delete coarse data array + excess data after interpolation
    del moonlight_array, interpolator, points, interpolated_flat
    gc.collect()
    
    # create lunar data array with proper structure
    moonlight_da = xr.DataArray(
        moonlight_highres,
        dims=["step", "y", "x"],
        coords={
            "valid_time": ("step", timesteps),
            "latitude": (("y", "x"), lats_2d),
            "longitude": (("y", "x"), lons_2d),
            "step": target_da['step'].data,
            "y": target_da.y.values,  
            "x": target_da.x.values},
        name="moonlight"
    )

    logger.info('Clipping Moon data to ConUSA boundary...')
    # let's clip the lunar data in WGS84
    USA_shp = gpd.read_file('scripts/utils/geo_ref_data/cb_2018_us_nation_5m.shp')
    conusa_mask = (-124.8, 24.4, -66.8, 49.4)
    conusa = gpd.clip(USA_shp, conusa_mask).to_crs("EPSG:4326")
    conus_boundary = conusa.geometry.union_all()

    # Create mask using lat/lon coordinates 
    lats = moonlight_da.latitude.values
    lons = moonlight_da.longitude.values
    mask = shapely.contains_xy(conus_boundary, lons.ravel(), lats.ravel()).reshape(lons.shape)

    # apply the mask, preserving all coordinate information
    moonlight_clipped = moonlight_da.where(mask)
    # remove unclipped array after masking
    del moonlight_da, mask
    gc.collect()

    # now, let's set the CRS and Transform
    moonlight_clipped.rio.write_crs(target_da.rio.crs, inplace=True)
    moonlight_clipped.rio.write_transform(target_da.rio.transform(), inplace=True)

    # Copy important attributes
    target_attrs = target_da.attrs.copy()
    target_attrs['description'] = 'Moon illumination fraction (0-1)'
    moonlight_clipped.attrs = target_attrs
    
    upload_zarr_dataset(moonlight_clipped, "processed-data/Moon_Dataset_Latest.zarr")
    logger.info('Done!')
    return moonlight_clipped
    
# preparing light pollution data for grade calculation
@flow(name='light-pollution-prep-subflow', log_prints=True)
def light_pollution_prep_subflow(bucket_name, target_da):
    """
    input:
        - bucket_name: cloud bucket where lp geotiff resides
        - target_da: NWS da with target lat/lon, step dim, 
                      and valid_time dim data
    """
    logger = logging_setup()
    logger.info('Subflow: prepping light pollution data for grading')
    
    # import High-Res Artificial Night Sky Brightness data from David Lorenz 
    lightpollution_da = load_tiff_from_R2(bucket_name,
                        "light-pollution-data/zenith_brightness_v22_2024_ConUSA.tif")
    lightpollution_da.load()
    
    # ensure it has the correct source CRS
    if lightpollution_da.rio.crs is None:
        lightpollution_da.rio.write_crs("EPSG:4326", inplace=True)
    
    # Clip to CONUS bounds in geographic coordinates
    lightpollution_clipped = lightpollution_da.rio.clip_box(
        minx=-126, miny=24, maxx=-66, maxy=50
    )
    # delete original data array after rough clip
    del lightpollution_da
    gc.collect()
    
    # Apply CONUS Mask
    logger.info('Clipping light pollution data to ConUSA boundary...')
    USA_shp = gpd.read_file('scripts/utils/geo_ref_data/cb_2018_us_nation_5m.shp')
    conusa_mask = (-124.8, 24.4, -66.8, 49.4)
    conusa = gpd.clip(USA_shp, conusa_mask).to_crs("EPSG:4326")

    # clip lp da using CONUS mask
    lp_clipped = lightpollution_clipped.rio.clip(conusa.geometry, 
                                     "EPSG:4326", 
                                     drop=False, 
                                     all_touched=True)
    # delete rough clipped data array after applying mask
    del lightpollution_clipped, USA_shp, conusa
    gc.collect()

    # set a variable name and select the first band of data
    lp_clipped = lp_clipped.isel(band=0).rename('light_pollution_normalized')

    # fixing coordinate names, so coords are lat/lon with dims y/x
    lp_clipped = lp_clipped.assign_coords({
        'longitude': ("x", lp_clipped.x.values),
        'latitude': ('y', lp_clipped.y.values)})

    # Get target's 2D lat/lon coordinates
    target_lats_2d = target_da.latitude.values
    target_lons_2d = target_da.longitude.values    

    # Interpolate to match NWS grid
    lp_lats = lp_clipped.latitude.values
    lp_lons = lp_clipped.longitude.values
    lp_data = lp_clipped.values
    
    interpolator = RegularGridInterpolator(
        (lp_lats, lp_lons),
        lp_data,
        method='linear',
        bounds_error=False,
        fill_value=np.nan
    )
    
    # Interpolate to target grid
    points = np.column_stack([target_lats_2d.ravel(), target_lons_2d.ravel()])
    lp_interp_flat = interpolator(points)
    lp_interp = lp_interp_flat.reshape(target_lats_2d.shape)
    # remove interpolation intermediate data, no longer needed
    del lp_clipped, interpolator, points, lp_interp_flat
    gc.collect()

    # calculate light pollution thresholds for normalization
    mag_thresholds, normalized_values = calc_LP_thresholds_task()
    # normalize artifificial night sky brightness from 0 to 1
    lp_interp_norm = normalize_light_pollution_task(lp_interp, 
                                                mag_thresholds, 
                                                normalized_values)

    # create data array of interp data w/ original brightness values
    lp_da_highres = xr.DataArray(
        lp_interp,
        dims=["y", "x"],
        coords={
            "latitude": (("y", "x"), target_lats_2d),
            "longitude": (("y", "x"), target_lons_2d),
            "y": target_da.y.values,
            "x": target_da.x.values
        },
        name="light_pollution"
    )
    # create data array of interp data w/ normalized brightness values
    lp_norm_highres = xr.DataArray(
        lp_interp_norm,
        dims=["y", "x"],
        coords={
            "latitude": (("y", "x"), target_lats_2d),
            "longitude": (("y", "x"), target_lons_2d),
            "y": target_da.y.values,
            "x": target_da.x.values
        },
        name="light_pollution_normalized"
    )

    # stack 2D LP arrays across a new "step" dimension
    step_size = target_da.sizes['step']
    # stacking the unaltered LP dataset
    lp_da_3d = xr.concat([lp_da_highres] * step_size, dim='step')
    lp_da_3d = lp_da_3d.assign_coords(
        step=target_da['step'],
        valid_time=target_da['valid_time'])
    # stacking the normalized LP dataset
    lp_norm_3d = xr.concat([lp_norm_highres] * step_size, dim='step')
    lp_norm_3d = lp_norm_3d.assign_coords(
        step=target_da['step'],
        valid_time=target_da['valid_time'])
    # delete 2D versions after stacking
    del lp_da_highres, lp_norm_highres, lp_interp, lp_interp_norm
    gc.collect()

    # merge both LP data arrays into an xarray dataset   
    lp_ds_3d = xr.merge([
        lp_da_3d,
        lp_norm_3d
    ], combine_attrs='drop_conflicts', compat='override')
    # remove individual 3D arrays after merge
    del lp_da_3d, lp_norm_3d
    gc.collect()

    # Copy spatial metadata from target
    lp_ds_3d['light_pollution'].rio.write_crs(target_da.rio.crs, inplace=True)
    lp_ds_3d['light_pollution'].rio.write_transform(target_da.rio.transform(), inplace=True)
    lp_ds_3d['light_pollution_normalized'].rio.write_crs(target_da.rio.crs, inplace=True)
    lp_ds_3d['light_pollution_normalized'].rio.write_transform(target_da.rio.transform(), inplace=True)

    # Copy essential GRIB attributes to both variables
    grib_attrs_to_copy = [
        'GRIB_DxInMetres', 'GRIB_DyInMetres', 'GRIB_gridType',
        'GRIB_LaDInDegrees', 'GRIB_Latin1InDegrees', 'GRIB_Latin2InDegrees',
        'GRIB_LoVInDegrees', 'GRIB_Nx', 'GRIB_Ny', 'GRIB_scanningMode'
    ]
    for attr in grib_attrs_to_copy:
        if attr in target_da.attrs:
            for var_name in lp_ds_3d.data_vars:
                lp_ds_3d[var_name].attrs[attr] = target_da.attrs[attr]
    # add description attribute
    lp_ds_3d['light_pollution'].attrs['description'] = 'Artificial night sky brightness in mag/arcsec² (22-17.8)'
    lp_ds_3d['light_pollution_normalized'].attrs['description'] = 'Normalized artificial night sky brightness (0-1)'

    upload_zarr_dataset(lp_ds_3d, "processed-data/LightPollution_Dataset_Latest.zarr")
    
    return lp_ds_3d['light_pollution_normalized']

# ----- FLOWS ----------------------------------------------------------------#
# ----- Stargazing Grade Calculation Flow -----
@flow(name="main-stargazing-calc-flow", log_prints=True)
def main_stargazing_calc_flow(skip_stargazing_tiles=False):
    # suppress Dask task logging
    import logging
    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.WARNING)

    # Keep attributes during all operations
    xr.set_options(keep_attrs=True)

    logger = logging_setup()
    logger.info('preparing meteorological and astronomical data...')
    clouds_da = cloud_cover_forecast_prep_subflow()
    precip_da = precip_forecast_prep_subflow()
   
    # ensuring that NWS datasets cover the same forecast datetimes
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
   
    moon_da = moon_data_prep_subflow(clouds_da)
    lp_da = light_pollution_prep_subflow("optimal-stargazing-locator",
                                         clouds_da)
    
    logger.info("evaluating each variable's effect on stargazing conditions...")
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

    # let's ensure each variable has the proper CRS and Transform
    for grade_da in [clouds_grades, precip_grades, lp_grades, moon_grades]:
        if grade_da.rio.crs is None:
            grade_da.rio.write_crs(clouds_da.rio.crs, inplace=True)
        if grade_da.rio.transform() is None:
            grade_da.rio.write_transform(clouds_da.rio.transform(), inplace=True)

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

    # merge into final dataset
    stargazing_ds = xr.merge([
        stargazing_grades.rename("grade_num"),
        precip_grades.rename("grade_precip"),
        clouds_grades.rename("grade_cloud"),
        lp_grades.rename("grade_lightpollution"),
        moon_grades.rename("grade_moon")
    ], combine_attrs='drop_conflicts', compat='override')

    # attach metadata
    stargazing_ds.attrs["legend"] = grade_legend
    stargazing_ds.attrs["description"] = "Weighted average grading of stargazing conditions"

    # mitigating sources for error 
    for var in stargazing_ds.data_vars:
        # ensure proper numeric dtype
        stargazing_ds[var] = stargazing_ds[var].astype('int8')  
        # remove old encoding->new metadata infers current state
        stargazing_ds[var].encoding.clear()
        # ensure crs and transform are preserved
        stargazing_ds[var].rio.write_crs(clouds_da.rio.crs, inplace=True)
        stargazing_ds[var].rio.write_transform(clouds_da.rio.transform(), inplace=True)
    # ensuring chunk alignment
    stargazing_ds = stargazing_ds.chunk(target_chunks)

    logger.info("uploading stargazing evaluation dataset to cloud...")
    upload_zarr_dataset(stargazing_ds, "processed-data/Stargazing_Dataset_Latest.zarr")

    logger.info('creating GIF of latest stargazing condition grades forecast')
    create_stargazing_gif(stargazing_ds['grade_num'],
                          'Stargazing Conditions Grades',
                          ['N/A','A+','A','B','C','D','F']) 
    
    # debugging step before generating tiles
    logger.info(f"Stargazing grades stats:," 
                f"min={float(stargazing_ds['grade_num'].min())},"
                f"max={float(stargazing_ds['grade_num'].max())},"
                f"shape={stargazing_ds['grade_num'].shape}")
    logger.info(f"CRS: {stargazing_ds['grade_num'].rio.crs}")
    logger.info(f"Transform: {stargazing_ds['grade_num'].rio.transform()}")
    logger.info(f"Has NaN values: {bool(np.any(np.isnan(stargazing_ds['grade_num'].values)))}")

    if skip_stargazing_tiles == False:
        logger.info(('generating stargazing grade tileset'))
        gen_tiles_task(stargazing_ds['grade_num'].assign_attrs((stargazing_ds.attrs | clouds_da.attrs)), 
                       "stargazing_grade", "data-layer-tiles/Stargazing_Tiles", 0.01, "gnuplot2_r",
                       vmin=-1, vmax=5, skip_tiles=skip_stargazing_tiles)
        
# ----- Preprocessing Precipitation data -----
@flow(name='precipitation-forecast-download-flow', log_prints=True)
def precipitation_forecast_flow(colormap="PuBuGn", skip_tiles=False):
    logger = logging_setup()
    logger.info('Flow: Retrieving Precipitation Data')
    ds = download_precip_task()
    logger.info('downloaded latest forecast')
    gif_boolean = create_gif_task(ds, colormap, "Precipitation Probability (%)", 
                                  "Precipitation Probability")
    if not gif_boolean:
        logger.error('GIF Creation Failed')
    logger.info('GIF saved to cloud')
    tile_boolean = gen_tiles_task(ds, "precip_probability", 
                                  "data-layer-tiles/PrecipProb_Tiles",
                                  0.01, colormap, vmin=0, vmax=100, 
                                  skip_tiles=skip_tiles)
    if not tile_boolean:
        logger.error('Tileset generation failed')
    logger.info('Preprocessing Precipitation Complete!')
    
# ----- Preprocessing Cloud Cover data -----
@flow(name='cloud-cover-forecast-download-flow', log_prints=True)
def cloud_cover_forecast_flow(colormap="YlGnBu", skip_tiles=False):
    logger = logging_setup()
    logger.info('Flow: Retrieving Cloud Cover Data')
    ds = download_sky_task()
    logger.info('downloaded latest forecast')
    gif_boolean = create_gif_task(ds, colormap, "Sky Covered by Clouds (%)", 
                                  "Cloud Coverage")
    if not gif_boolean:
        logger.error('GIF Creation Failed')
    logger.info('GIF saved to cloud')
    tile_boolean = gen_tiles_task(ds, "cloud_coverage", 
                                  "data-layer-tiles/SkyCover_Tiles",
                                  0.05, colormap, vmin=0, vmax=100,
                                  skip_tiles=skip_tiles)
    if not tile_boolean:
        logger.error('Tileset generation failed')
    logger.info('Preprocessing Cloud Cover Complete!')

# ----- Stargazing Grade Plot Creation Flow (Testing) -----
@flow(name='stargazing-grade-gif-test-flow', log_prints=True)
def test_stargazing_gif_flow():
    logger = logging_setup()
    # import the stargazing grade data array
    stargazing_ds = load_zarr_from_R2('optimal-stargazing-locator', "processed-data/Stargazing_Dataset_Latest.zarr")

    create_stargazing_gif(stargazing_ds['grade_num'],
                          'Stargazing Conditions Grades',
                          ['N/A','A+','A','B','C','D','F']) 
    logger.info('Done!')

# ----- Simplified version of the Stargazing Grade Calculation Flow -----
@flow(name="simplified-stargazing-calc-flow", log_prints=True)
def simplified_stargazing_calc_flow(skip_stargazing_tiles=False):
    """
    This is a cohesive and simplified version of the Stargazing Grade Calculation flow, wherein:
    - The latest precipitation and cloud cover are downloaded from the NWS
    - All pertinent meteorological and astronomical datasets are prepared for grade evaluations
    - Stargazing Grades are calculated at the spatiotemporal level
    - Tiles and GIFs are created for the Stargazing Grade dataset ONLY
    """
    # suppress Dask task logging
    import logging
    logging.getLogger("distributed").setLevel(logging.WARNING)
    logging.getLogger("dask").setLevel(logging.WARNING)

    # Keep attributes during all operations
    xr.set_options(keep_attrs=True)

    logger = logging_setup()
    
    logger.info('Flow: Retrieving Precipitation Data')
    precip_da = download_precip_task()
    
    logger.info('Preparing precip data for grade calculations')
    # normalize the precipitation data on a scale (0=no rain, 1=100% Showers)
    precip_da_norm = precip_da / 100.0
    
    # convert longitudes from 0–360 to -180–180
    precip_da_norm = precip_da_norm.assign_coords(
        longitude=((precip_da_norm.longitude + 180) % 360) - 180)
    # ensure the proper CRS metadata is preserved
    if precip_da_norm.rio.crs is None:
        ndfd_proj4 = (
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
            "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
        )
        precip_da_norm.rio.write_crs(ndfd_proj4, inplace=True)
        
        # Set the spatial transform (from your tile generation code)
        dx = precip_da_norm.attrs["GRIB_DxInMetres"]
        dy = precip_da_norm.attrs["GRIB_DyInMetres"]
        minx = -2764474.3507319926
        maxy = 3232111.7107923944
        transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)
        precip_da_norm.rio.write_transform(transform, inplace=True)
    del precip_da  # remove original data array
    gc.collect() # delete no longer needed data

    logger.info('Flow: Retrieving Cloud Cover Data')
    clouds_da = download_sky_task()
    
    logger.info('Preparing cloud cover data for grade calculations')
    # normalize the sky coverage data on a 0-1 scale (0=blue skies, 1=totally cloudy)
    clouds_da_norm = clouds_da/100.0
    # convert longitudes from 0–360 to -180–180
    clouds_da_norm = clouds_da_norm.assign_coords(
        longitude=((clouds_da_norm.longitude + 180) % 360) - 180)
    # ensure the proper CRS metadata is preserved
    if clouds_da_norm.rio.crs is None:
        ndfd_proj4 = (
            "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 "
            "+lon_0=-95 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs"
        )
        clouds_da_norm.rio.write_crs(ndfd_proj4, inplace=True)
        
        # Set the spatial transform
        dx = clouds_da_norm.attrs["GRIB_DxInMetres"]
        dy = clouds_da_norm.attrs["GRIB_DyInMetres"]
        minx = -2764474.3507319926
        maxy = 3232111.7107923944
        transform = affine.Affine(dx, 0, minx, 0, -dy, maxy)
        clouds_da_norm.rio.write_transform(transform, inplace=True)
    del clouds_da  # remove original data array
    gc.collect() # delete no longer needed data

    logger.info('preparing lunar and light pollution data...')
    # ensuring that NWS datasets cover the same forecast datetimes
    precip_da = precip_da_norm[np.isin(precip_da_norm['valid_time'].values,
                                             clouds_da_norm['valid_time'].values)]
    clouds_da = clouds_da_norm[np.isin(clouds_da_norm['valid_time'].values,
                                                 precip_da_norm['valid_time'].values)]
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
    # clean up of unfiltered normalized data arrays
    del precip_da_norm, clouds_da_norm
    gc.collect()
   
    moon_da = moon_data_prep_subflow(clouds_da)
    lp_da = light_pollution_prep_subflow("optimal-stargazing-locator",
                                         clouds_da)
    
    logger.info("evaluating each variable's effect on stargazing conditions...")
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
    # cleanup data arrays after grading
    del precip_da, lp_da, moon_da # clouds attrs needed later on
    gc.collect()

    # let's ensure each variable has the proper CRS and Transform
    for grade_da in [clouds_grades, precip_grades, lp_grades, moon_grades]:
        if grade_da.rio.crs is None:
            grade_da.rio.write_crs(clouds_da.rio.crs, inplace=True)
        if grade_da.rio.transform() is None:
            grade_da.rio.write_transform(clouds_da.rio.transform(), inplace=True)    

    logger.info('calculating spatiotemporal stargazing grades')
    # variable weights
    w_precip = 0.5
    w_cloud = 0.85
    w_lp = 1.0
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

    # merge into final dataset
    stargazing_ds = xr.merge([
        stargazing_grades.rename("grade_num"),
        precip_grades.rename("grade_precip"),
        clouds_grades.rename("grade_cloud"),
        lp_grades.rename("grade_lightpollution"),
        moon_grades.rename("grade_moon")
    ], combine_attrs='drop_conflicts', compat='override')
    # remove all individual grade arrays, no longer needed
    del stargazing_grades, precip_grades, clouds_grades, lp_grades, moon_grades
    gc.collect()

    # attach metadata
    stargazing_ds.attrs["legend"] = grade_legend
    stargazing_ds.attrs["description"] = "Weighted average grading of stargazing conditions"

    # mitigating sources for error 
    for var in stargazing_ds.data_vars:
        # ensure proper numeric dtype
        stargazing_ds[var] = stargazing_ds[var].astype('int8')  
        # remove old encoding->new metadata infers current state
        stargazing_ds[var].encoding.clear()
        # ensure crs and transform are preserved
        stargazing_ds[var].rio.write_crs(clouds_da.rio.crs, inplace=True)
        stargazing_ds[var].rio.write_transform(clouds_da.rio.transform(), inplace=True)
    # ensuring chunk alignment
    stargazing_ds = stargazing_ds.chunk(target_chunks)
    gc.collect() # delete no longer needed data

    logger.info("uploading stargazing evaluation dataset to cloud...")
    upload_zarr_dataset(stargazing_ds, "processed-data/Stargazing_Dataset_Latest.zarr")
    gc.collect() # delete no longer needed data

    logger.info('creating GIF of latest stargazing condition grades forecast')
    create_stargazing_gif(stargazing_ds['grade_num'],
                          'Stargazing Grade',
                          ['A+','A','B','C','D','F']) 
    
    # debugging step before generating tiles
    logger.info(f"Stargazing grades stats:," 
                f"min={float(stargazing_ds['grade_num'].min())},"
                f"max={float(stargazing_ds['grade_num'].max())},"
                f"shape={stargazing_ds['grade_num'].shape}")
    logger.info(f"CRS: {stargazing_ds['grade_num'].rio.crs}")
    logger.info(f"Transform: {stargazing_ds['grade_num'].rio.transform()}")
    logger.info(f"Has NaN values: {bool(np.any(np.isnan(stargazing_ds['grade_num'].values)))}")
    # aggressive cleanup before tile generation
    gc.collect()
    time.sleep(3)  # brief pause to ensure cleanup completes
    log_memory_usage('before generating stargazing tiles')
    
    if skip_stargazing_tiles == False:
        logger.info(('generating stargazing grade tileset'))
        gen_tiles_task(stargazing_ds['grade_num'].assign_attrs((stargazing_ds.attrs | clouds_da.attrs)), 
                       "stargazing_grade", "data-layer-tiles/Stargazing_Tiles", 0.01, cmap=None,
                       vmin=0, vmax=5, skip_tiles=skip_stargazing_tiles)
        
# for script execution
if __name__ == "__main__":
    simplified_stargazing_calc_flow()