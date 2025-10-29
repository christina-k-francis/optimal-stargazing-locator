"""
Functions that implement and moderate logging throughout script execution.
"""

import os
import psutil
import logging
import warnings
from prefect import get_run_logger

def logging_setup(silence_packages: list[str] = None):
    """
    description:
        this FX is a boilerplate for setting up logging at the top 
        of all flows, subflows, and tasks that require it.
    """
    logger = get_run_logger()

    # redirect all warnings to the logger
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", category=UserWarning)

    # silence noisy packages
    silence_packages = silence_packages or ["boto3"]
    for pkg in silence_packages:
        logging.getLogger(pkg).setLevel(logging.WARNING)

    return logger

def log_memory_usage(stage: str):
    """
    description:
        Logs the RAM usage (RSS Memory) at its position in the script.
    
    input:
        stage (str): A label for the stage in execution where memory is measured.
    """
    logger = get_run_logger()
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    logger.info(f"[MEMORY] RSS memory usage at {stage}: {mem:.2f} MB")