"""
This is the main orechestration flow for calculating stargazing grades
"""

from prefect import flow
from flows import (
    precipitation_forecast_flow,
    cloud_cover_forecast_flow,
    main_stargazing_calc_flow,
)
from scripts.utils.logging_tools import logging_setup, log_memory_usage


@flow(name="stargazing-orchestration-flow", log_prints=True)
def stargazing_orchestration_flow():
    logger = logging_setup()
    log_memory_usage("at the start of the main orchestration script")

    # run downloading/preprocessing flows as parallel tasks
    sky_future = cloud_cover_forecast_flow(return_state=True)
    precip_future = precipitation_forecast_flow(return_state=True)

    logger.info("Triggered sky coverage and precipitation flows concurrently.")

    # check if they completed successfully
    if sky_future.is_failed():
        logger.error("Sky coverage flow failed!")
        raise Exception("Sky coverage flow failed")
    
    if precip_future.is_failed():
        logger.error("Precipitation flow failed!")
        raise Exception("Precipitation flow failed")

    logger.info("both preprocessing flows completed successfully.")
    log_memory_usage("before running the stargazing grade calc flow")

    # Now we'll run the main Stargazing Grade calculator flow
    main_stargazing_calc_flow()
    logger.info("main stargazing calc flow completed.")