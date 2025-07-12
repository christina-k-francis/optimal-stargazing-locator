"""
Main processing script for the temperature dataset.
Generating a GIF of the latest forecast, and a tileset.
Both are uploaded to the cloud.
"""

from nws_average_temperature_download import get_temperature
from utils.gif_tools import create_nws_temp_gif
from utils.tile_tools import generate_tiles_from_zarr
from utils.memory_logger import log_memory_usage
import gc

def main():
    log_memory_usage("Start of Temperature Processing Script")
    ds = get_temperature()
    create_nws_temp_gif(ds, "RdYlBu_r", "Temperature in Fahrenheit and Celsius")
    generate_tiles_from_zarr(ds, "temperature", "data-layer-tiles/Temp_Tiles", 0.005)
    del ds
    gc.collect()
    log_memory_usage("Successful End of Temperature Processing Script!")

main()
