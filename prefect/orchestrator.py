"""
This is the main orechestration flow for calculating stargazing grades
"""

from prefect import flow, get_run_logger
from flows import (
    precipitation_forecast_flow,
    cloud_cover_forecast_flow,
    main_stargazing_calc_flow,
)


@flow(name="stargazing-orchestration-flow", log_prints=True)
def stargazing_orchestration_flow():
    logger = get_run_logger()

    # Run preprocessing flows as parallel tasks
    sky_future = cloud_cover_forecast_flow.submit()
    precip_future = precipitation_forecast_flow.submit()

    logger.info("Triggered sky coverage and precipitation flows concurrently.")

    # Wait for both to finish before continuing
    sky_future.result()
    precip_future.result()

    logger.info("both preprocessing flows completed successfully.")

    # Now we'll run the main Stargazing flow
    main_stargazing_calc_flow()
    logger.info("main stargazing calc flow completed.")