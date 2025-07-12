"""
Main processing script for the precipitation probability dataset.
Generating a GIF of the latest forecast, and a tileset.
Both are uploaded to the cloud.
"""

from nws_precipitation_probability_download import get_precip_probability
from pypalettes import load_cmap
from utils.gif_tools import create_nws_gif
from utils.tile_tools import generate_tiles_from_zarr
from utils.memory_logger import log_memory_usage
import gc

def main():
    log_memory_usage("Start of Precipitation Prob. Processing Script")
    ds = get_precip_probability()
    create_nws_gif(ds, load_cmap("LightBluetoDarkBlue_7"), "Precipitation Probability (%)",
                    "Precipitation Probability")
    generate_tiles_from_zarr(ds, "precip_probability", "data-layer-tiles/PrecipProb_Tiles", 0.005)
    del ds
    gc.collect()
    log_memory_usage('Successful End of Precipitation Processing Script!')

main()
