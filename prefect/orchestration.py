"""
This is the main orechestration flow for calculating stargazing grades
"""

from prefect import flow
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    logger.info("Executing sky coverage and precipitation flows concurrently.")
    # run downloading/preprocessing flows in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both flows to run in parallel
        sky_flow = executor.submit(cloud_cover_forecast_flow, skip_tiles=True)
        precip_flow = executor.submit(precipitation_forecast_flow, skip_tiles=True)
        
        # Wait for both to complete and collect results
        flows = {
            'cloud': sky_flow,
            'precip': precip_flow}
        
        results, errors = {}, {}
        
        for name, flow in flows.items():
            try:
                results[name] = flow.result() # waiting until flow is complete
                logger.info(f"{name} flow completed successfully")
            except Exception as e:
                errors[name] = str(e)
                logger.error(f"{name} flow failed: {e}")

    # check if the flows completed successfully
    if 'cloud' in errors:
        raise Exception(f"Sky coverage flow failed: {errors['cloud']}")
    
    if 'precip' in errors:
        raise Exception(f"Precipitation flow failed: {errors['precip']}")

    logger.info("both preprocessing flows completed successfully.")
    log_memory_usage("before running the stargazing grade calc flow")

    # Now we'll run the main Stargazing Grade calculator flow
    main_stargazing_calc_flow(skip_stargazing_tiles=True)
    logger.info("Orchestration Complete!")