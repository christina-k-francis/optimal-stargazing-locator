# tile_server.py
"""

Created on Sat June 28 18:24:00 2025

@author: Christina
"""
###
"""
    This script imports zarr data from the cloud and converts
    it into imagery tiles for serving to Mapbox. There, they'll 
    be displayed as map layers.
"""
###
import os
from flask import Flask, send_file, abort, jsonify
import tempfile
import logging
from storage3 import create_client
from pathlib import Path
import threading
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Supabase setup
SUPABASE_URL = "https://rndqicxdlisfpxfeoeer.supabase.co"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
if not SUPABASE_KEY:
    logging.error("Missing SUPABASE_KEY in environment variables.")
    raise EnvironmentError("SUPABASE_KEY is required but not set.")

storage = create_client(f"{SUPABASE_URL}/storage/v1",
                        {"Authorization": f"Bearer {SUPABASE_KEY}"},
                        is_async=False)

BUCKET_NAME = "maps"

# Local cache setup
CACHE_DIR = Path("tile_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Tile freshness threshold (in seconds)
CACHE_EXPIRY_SECONDS = 12 * 3600  # 12 hours

# Map layer to Supabase folder mapping
LAYER_PATHS = {
    "SkyCover_Tiles": "data-layer-tiles/SkyCover_Tiles",
    "PrecipProb_Tiles": "data-layer-tiles/PrecipProb_Tiles",
    "Temp_Tiles": "data-layer-tiles/Temp_Tiles",
    "Stargazing_Tiles": "data-layer-tiles/Stargazing_Tiles",
}


@app.route('/tiles/<layer>/<int:z>/<int:x>/<int:y>.png')
def serve_tile(layer, z, x, y):
    if layer not in LAYER_PATHS:
        abort(404, description="Layer not found")

    local_path = CACHE_DIR / layer / str(z) / str(x) / f"{y}.png"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return send_file(local_path, mimetype='image/png')

    supabase_path = f"{LAYER_PATHS[layer]}/{z}/{x}/{y}.png"
    try:
        tile_data = storage.from_(BUCKET_NAME).download(supabase_path)
        if tile_data is None:
            logging.warning(f"Tile not found at {supabase_path}")
            abort(404, description="Tile not found")

        with open(local_path, "wb") as f:
            f.write(tile_data)

        return send_file(local_path, mimetype='image/png')

    except Exception as e:
        logging.error(f"Error fetching tile {supabase_path}: {e}")
        abort(404, description="Tile not found")


@app.route('/health')
def health_check():
    """Basic health check endpoint for Render monitoring."""
    return jsonify({"status": "ok"}), 200


def periodic_cache_cleanup():
    """Deletes cached tiles older than CACHE_EXPIRY_SECONDS every hour."""
    while True:
        now = time.time()
        deleted_files = 0

        for tile_path in CACHE_DIR.rglob("*.png"):
            if tile_path.is_file():
                file_age = now - tile_path.stat().st_mtime
                if file_age > CACHE_EXPIRY_SECONDS:
                    try:
                        tile_path.unlink()
                        deleted_files += 1
                    except Exception as e:
                        logging.error(f"Failed to delete {tile_path}: {e}")

        if deleted_files > 0:
            logging.info(f"Cache cleanup complete. Deleted {deleted_files} expired tiles.")
        else:
            logging.info("Cache cleanup complete. No expired tiles found.")

        time.sleep(3600)  # Wait 1 hour before next cleanup


if __name__ == '__main__':
    # Start background cache cleanup thread
    cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
    cleanup_thread.start()

    app.run(host='0.0.0.0', port=5000)
