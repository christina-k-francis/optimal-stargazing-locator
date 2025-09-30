"""
This is the main orechestration flow for calculating stargazing grades
"""

from prefect import flow, get_run_logger
from flows import (
    process_sky_coverage_flow,
    process_precipitation_flow,
    main_stargazing_flow,
)


@flow(name="stargazing-orchestration-flow", log_prints=True)
def stargazing_orchestration_flow():
    logger = get_run_logger()

    # Run preprocessing flows as parallel tasks
    sky_future = process_sky_coverage_flow.submit()
    precip_future = process_precipitation_flow.submit()

    logger.info("Triggered Sky Coverage and Precipitation flows concurrently.")

    # Wait for both to finish before continuing
    sky_future.result()
    precip_future.result()

    logger.info("Both preprocessing flows completed successfully.")

    # Now we'll run the main Stargazing flow
    main_stargazing_flow()
    logger.info("Main stargazing flow completed.")