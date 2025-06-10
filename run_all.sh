#!/bin/bash
set -e  # Exit if any command fails

echo "Running NWS data download script..."
python -m scripts.main_download_nws

echo "Running Stargazing Index script..."
python -m scripts.main_stargazing_index
