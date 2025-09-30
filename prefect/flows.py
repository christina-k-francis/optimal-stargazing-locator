"""
A script for organizing overarching prefect flows
"""


from prefect import flow, task
import gc

# import custom functions/modules
from scripts.utils.gif_tools import create_nws_gif, create_stargazing_gif
from scripts.utils.tile_tools import generate_tiles_from_zarr, generate_stargazing_tiles
from scripts.utils.memory_logger import log_memory_usage

# ---- small tasks ----
# Downloading latest NWS sky coverage data
@task(log_prints=True, retries=1)
def download_sky_task():
    from scripts.nws_sky_coverage_download import get_sky_coverage
    ds = get_sky_coverage()
    return ds
# Downloading latest NWS precip. probability data
@task(log_prints=True, retries=1)
def download_precip_task():
    from scripts.nws_precipitation_probability_download import get_precipitation_prob
    ds = get_precipitation_prob()
    return ds

# create a gif of NWS data variables
@task(log_prints=True, retries=1)
def gif_task(ds, cmap, title, shortname):
    create_nws_gif(ds, cmap, title, shortname)
    return True

# generate tiles from NWS data variables
@task(log_prints=True, retries=1)
def tiles_task(ds, layer_name, out_dir, res, cmap):
    generate_tiles_from_zarr(ds, layer_name, out_dir, res, cmap)
    return True

# A thin wrapper for your heavy main_stargazing_calc
@task(log_prints=True, retry_delay_seconds=60, retries=0)
def main_stargazing_task():
    # import inside the task to avoid requiring the module at import-time on the deploy server
    from scripts.main_stargazing_calc import main as stargazing_main
    stargazing_main()
    return True

# ---- flows ----
# pre-processing sky coverage data
@flow(name="process-sky-coverage-flow", log_prints=True)
def process_sky_coverage_flow():
    log_memory_usage("Start of Sky Cover Processing Script")
    ds = download_sky_task()
    log_memory_usage("After importing sky cover dataset")
    gif_task(ds, "bone", "Percentage of Sky Covered by Clouds", "Cloud Coverage")
    log_memory_usage("After creating sky cover gif")
    tiles_task(ds, "cloud_coverage", "data-layer-tiles/SkyCover_Tiles", 0.005, "bone")
    log_memory_usage("After generating sky cover tileset")
    # cleanup
    ds = None
    gc.collect()
    log_memory_usage("Successful End of Sky Cover Processing Script!")

# pre-processing precipitation probability data
@flow(name="process-precipitation-flow", log_prints=True)
def process_precipitation_flow():
    log_memory_usage("Start of Precip Processing Script")
    ds = download_precip_task()
    log_memory_usage("After importing precip dataset")
    gif_task(ds, "viridis", "Precipitation Probability", "Precip")
    tiles_task(ds, "precip_prob", "data-layer-tiles/Precip_Tiles", 0.005, "viridis")
    ds = None
    gc.collect()
    log_memory_usage("End Precip Script")

#  calculating stargazing grades
@flow(name="main-stargazing-flow", log_prints=True)
def main_stargazing_flow():
    main_stargazing_task()
