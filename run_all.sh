#!/bin/bash
set -e  # Exit if any command fails

echo "Running NWS data downloader script..."
#python -m scripts.main_nws_download

echo "Running Stargazing Index calculator script..."
python -m scripts.main_stargazing_index
