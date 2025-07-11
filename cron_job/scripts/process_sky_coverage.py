"""
Main processing script for the sky cover dataset.
Generating a GIF of the latest forecast, and a tileset.
Both are uploaded to the cloud.
"""

import xarray as xr
from pypalettes import load_cmap
import gc
from nws_sky_coverage_download import get_sky_coverage
from pypalettes import load_cmap
from utils.gif_tools import create_nws_gif
from utils.tile_tools import generate_tiles_from_zarr
from utils.memory_logger import log_memory_usage

def main():
    log_memory_usage("Start of Sky Cover Processing Script")
    ds = get_sky_coverage()
    log_memory_usage("After importing sky cover dataset")
    create_nws_gif(ds, load_cmap("Bmsurface"), "Percentage of Sky Covered by Clouds", "Cloud Coverage")
    log_memory_usage("After creating sky cover gif")
    generate_tiles_from_zarr(ds, "cloud_coverage", "data-layer-tiles/SkyCover_Tiles", 0.025)
    log_memory_usage("After generating sky cover tileset")
    del ds
    gc.collect()
    log_memory_usage("End of Sky Cover Processing Script")

main()
