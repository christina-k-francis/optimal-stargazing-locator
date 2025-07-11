"""
Main processing script for auxillary datasets that don't go into
Stargazing Evaluations, nor are served as map layers.
Outputs are uploaded to the cloud.
"""

from nws_relative_humidity_download import get_relhum_percent
from nws_wind_speed_and_direction_download import get_wind_speed_direction
from utils.gif_tools import create_nws_gif
from utils.memory_logger import log_memory_usage
import gc

def main():
    log_memory_usage("Start Relative Humidity")
    rh_ds = get_relhum_percent()
    create_nws_gif(rh_ds, "pink_r", "Relative Humidity (%)", "Relative Humidity")
    del rh_ds
    gc.collect()

    log_memory_usage("Start Wind Speed/Dir")
    wind_ds = get_wind_speed_direction()
    # Not yet using; no plots or tiles needed
    del wind_ds
    gc.collect()

main()
