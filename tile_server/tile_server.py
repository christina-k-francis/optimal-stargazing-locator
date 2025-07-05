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
from fastapi import FastAPI, Response, HTTPException
import logging
from storage3 import create_client
from pathlib import Path
import threading
import time
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Supabase storage configuration
SUPABASE_URL = "https://rndqicxdlisfpxfeoeer.supabase.co"
BUCKET_NAME = "maps"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')
if not SUPABASE_KEY:
    logger.error("Missing SUPABASE_KEY in environment variables.")
    raise EnvironmentError("SUPABASE_KEY is required but not set.")
blank_tile_url = "https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/maps/data-layer-tiles/blank_tile_256x256.png"

storage = create_client(f"{SUPABASE_URL}/storage/v1",
                        {"Authorization": f"Bearer {SUPABASE_KEY}"},
                        is_async=False)

# Local cache setup
CACHE_DIR = Path("tile_cache")
CACHE_DIR.mkdir(exist_ok=True)

CACHE_EXPIRY_SECONDS = 12 * 3600  # 12 hours

LAYER_PATHS = {
    "SkyCover_Tiles": "data-layer-tiles/SkyCover_Tiles",
    "PrecipProb_Tiles": "data-layer-tiles/PrecipProb_Tiles",
    "Temp_Tiles": "data-layer-tiles/Temp_Tiles",
    "Stargazing_Tiles": "data-layer-tiles/Stargazing_Tiles",
}


@app.get("/tiles/{layer}/{timestamp}/{z}/{x}/{y}.png")
def get_tile(layer: str, timestamp: str, z: int, x: int, y: int):
    """Serve tile from cache or Supabase, fallback to blank tile if missing."""
    if layer not in LAYER_PATHS:
        raise HTTPException(status_code=404, detail="Layer not found")

    local_path = CACHE_DIR / layer / timestamp / str(z) / str(x) / f"{y}.png"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return Response(content=local_path.read_bytes(), media_type="image/png")

    supabase_path = f"{LAYER_PATHS[layer]}/{timestamp}/{z}/{x}/{y}.png"
    try:
        tile_data = storage.from_(BUCKET_NAME).download(supabase_path)
        if tile_data:
            with open(local_path, "wb") as f:
                f.write(tile_data)
            return Response(content=tile_data, media_type="image/png")
        else:
            raise Exception("Tile not found in Supabase")

    except Exception as e:
        logger.warning(f"Tile missing ({supabase_path}): {e}")
        blank_tile = requests.get(blank_tile_url)
        if blank_tile.status_code == 200:
            return Response(content=blank_tile.content, media_type="image/png")
        else:
            logger.error(f"Failed to fetch blank tile. Status code: {blank_tile.status_code}")
            raise HTTPException(status_code=500, detail="Tile unavailable")


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
                        logger.error(f"Failed to delete {tile_path}: {e}")

        if deleted_files > 0:
            logger.info(f"Cache cleanup complete. Deleted {deleted_files} expired tiles.")
        time.sleep(3600)


if __name__ == '__main__':
    cleanup_thread = threading.Thread(target=periodic_cache_cleanup, daemon=True)
    cleanup_thread.start()
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)