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
import logging
import threading
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from storage3 import create_client

# --- Configuration ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS setup
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing/dev, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase storage configuration
SUPABASE_URL = "https://rndqicxdlisfpxfeoeer.supabase.co"
BUCKET_NAME = "maps"
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

if not SUPABASE_KEY:
    logger.error("Missing SUPABASE_KEY in environment variables.")
    raise EnvironmentError("SUPABASE_KEY is required but not set.")

storage = create_client(f"{SUPABASE_URL}/storage/v1",
                        {"Authorization": f"Bearer {SUPABASE_KEY}"},
                        is_async=False)

# Map layers mapped to supabase folders
LAYER_PATHS = {
    "SkyCover_Tiles": "data-layer-tiles/SkyCover_Tiles",
    "PrecipProb_Tiles": "data-layer-tiles/PrecipProb_Tiles",
    "Temp_Tiles": "data-layer-tiles/Temp_Tiles",
    "Stargazing_Tiles": "data-layer-tiles/Stargazing_Tiles",
    "LightPollution_Tiles": "light-pollution-data/zenith_ConUSA_colored_tiles",
}

# blank tile configuration
blank_tile_url = "https://rndqicxdlisfpxfeoeer.supabase.co/storage/v1/object/public/maps/data-layer-tiles/blank_tile_256x256.png"

# Local cache setup
CACHE_DIR = Path("tile_cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_SECONDS = 12 * 3600  # 12 hours

# --- Tile Serving Endpoint --------------------------------------------------------------------
@app.get("/tiles/{layer}/{timestamp}/{z}/{x}/{y}.png")
async def get_tile(layer: str, timestamp: str, z: int, x: int, y: int):
    """Serve tile from cache or Supabase, flipping y to match Slippy Map."""

    if layer not in LAYER_PATHS:
        logger.warning(f"Invalid layer requested: {layer}")
        return Response(status_code=404, content="Layer not found")

    # Flip y value from TMS to Slippy format
    slippy_y = (2 ** z) - 1 - y

    # Local cache path
    local_path = CACHE_DIR / layer / timestamp / str(z) / str(x) / f"{slippy_y}.png"
    local_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "Cache-Control": "public, max-age=604800",  # Cache for 7 days
        "Content-Type": "image/png"
    }
    
    if local_path.exists():
        return StreamingResponse(open(local_path, "rb"), headers=headers)

    if layer == "LightPollution_Tiles" and timestamp == "static":
        supabase_path = f"{LAYER_PATHS[layer]}/{z}/{x}/{slippy_y}.png"
    else:
        supabase_path = f"{LAYER_PATHS[layer]}/{timestamp}/{z}/{x}/{slippy_y}.png"

    try:
        tile_data = storage.from_(BUCKET_NAME).download(supabase_path)
        if tile_data:
            with open(local_path, "wb") as f:
                f.write(tile_data)
            return StreamingResponse(open(local_path, "rb"), headers=headers)
    except Exception as e:
        logger.warning(f"Tile missing/error ({supabase_path}): {e}")

    # Serve blank tile in place of missing data (also cache it locally)
    return StreamingResponse(
    open(blank_tile_path, "rb"),
    headers={
        "Cache-Control": "public, max-age=604800",
        "Content-Type": "image/png"
    }
)

@app.head("/tiles/{layer}/{ts}/{z}/{x}/{y}.png")
async def head_tile(layer: str, ts: str, z: int, x: int, y: int):
    """Check tile existence in cache or cloud, with y-flip for Slippy Map."""
    if layer not in LAYER_PATHS:
        return Response(status_code=404)

    slippy_y = (2 ** z) - 1 - y
    # local cache path
    local_path = CACHE_DIR / layer / ts / str(z) / str(x) / f"{slippy_y}.png"

    if local_path.exists():
        return Response(status_code=200)
    # Attempt to check Supabase without downloading full file
    if layer == "LightPollution_Tiles" and ts == "static":
        supabase_path = f"{LAYER_PATHS[layer]}/{z}/{x}/{slippy_y}.png"
    else:
        supabase_path = f"{LAYER_PATHS[layer]}/{ts}/{z}/{x}/{slippy_y}.png"

    try:
        # Attempt to download metadata (HEAD isn't supported directly by Supabase Storage3)
        tile_data = storage.from_(BUCKET_NAME).download(supabase_path)
        if tile_data:
            return Response(status_code=200)
    except Exception as e:
        logger.info(f"Tile not found in Supabase: {supabase_path} | {e}")

    return Response(status_code=404)

def serve_blank_tile(cache_path: Path):
    """Fetches and serves the blank transparent tile """
    try:
        # Cache blank tile locally after first use
        blank_tile_path = CACHE_DIR / "blank_tile.png"
        if not blank_tile_path.exists():
            with httpx.stream("GET", blank_tile_url) as r:
                r.raise_for_status()
                with open(blank_tile_path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
            logger.info("Blank tile downloaded and cached locally.")

        # Copy blank tile to requested cache path
        cache_path.write_bytes(blank_tile_path.read_bytes())

        return StreamingResponse(open(blank_tile_path, "rb"), media_type="image/png")        
    except Exception as e:
        logger.error(f"Failed to fetch blank tile: {e}")
        return Response(status_code=500, content="Tile and fallback missing")


# --- Health Check ----------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Cache Cleanup Thread ----------------------------------------------------
def periodic_cache_cleanup():
    """Deletes cached tiles older than CACHE_EXPIRY_SECONDS."""
    while True:
        now = time.time()
        deleted = 0

        for tile in CACHE_DIR.rglob("*.png"):
            if tile.is_file():
                age = now - tile.stat().st_mtime
                if age > CACHE_EXPIRY_SECONDS:
                    try:
                        tile.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {tile}: {e}")

        logger.info(f"Cache cleanup complete. Deleted {deleted} expired tiles.")
        time.sleep(3600)  # Run hourly


# --- Main Entry Point -------------------------------------------------------
if __name__ == '__main__':
    threading.Thread(target=periodic_cache_cleanup, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
