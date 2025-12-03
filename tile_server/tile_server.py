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
import io
import os
import logging
import threading
import time
import s3fs
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Stargazing Web App Tile Server')

# CORS setup
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For testing/dev, allow all origins
    allow_credentials=True,
    allow_methods=["GET","HEAD","OPTIONS"],
    allow_headers=["*"],
)

# Cloudflare R2 (S3-compatible) Storage Configuration
account_id = os.environ["R2_ACCOUNT_ID"]
access_key = os.environ["R2_ACCESS_KEY"]
secret_key = os.environ["R2_SECRET_KEY"]

fs = s3fs.S3FileSystem(
    key=access_key,
    secret=secret_key,
    client_kwargs={"endpoint_url": f"https://{account_id}.r2.cloudflarestorage.com"}
)

BUCKET = "optimal-stargazing-locator"

# Mapping to Files in Bucket on Cloud
LAYER_PATHS = {
    "SkyCover_Tiles": "data-layer-tiles/SkyCover_Tiles",
    "PrecipProb_Tiles": "data-layer-tiles/PrecipProb_Tiles",
    "Temp_Tiles": "data-layer-tiles/Temp_Tiles",
    "Stargazing_Tiles": "data-layer-tiles/Stargazing_Tiles",
    "LightPollution_Tiles": "light-pollution-data/zenith_ConUSA_colored_tiles",
}

LEGEND_PATHS = {
        "Temp_Dark.png": "plots/Temp_Legend_Dark.png",
        "Temp_Light.png": "plots/Temp_Legend_Light.png",
        "Stargazing_Dark.png": "plots/Stargazing_Legend_Dark.png",
        "Stargazing_Light.png": "plots/Stargazing_Legend_Light.png",
        "SkyCover_Dark.png": "plots/CloudCover_Legend_Dark.png",
        "SkyCover_Light.png": "plots/CloudCover_Legend_Light.png",
        "PrecipProb_Dark.png": "plots/PrecipProb_Legend_Dark.png",
        "PrecipProb_Light.png": "plots/PrecipProb_Legend_Light.png",
        "LightPollution_Dark.png": "plots/LightPollution_Legend_Dark.png",
        "LightPollution_Light.png": "plots/LightPollution_Legend_Light.png"
    }

GIF_PATHS = {
    "Stargazing_Latest.gif": "plots/Stargazing_Dataset_Latest.gif",
    "PrecipProb_Latest.gif": "plots/Precipitation Probability_Latest.gif",
    "CloudCover_Latest.gif": "plots/Cloud Coverage_Latest.gif"
}

# blank tile configuration
blank_tile_key = "data-layer-tiles/blank_tile_256x256.png"

# Local cache setup
CACHE_DIR = Path("tile_cache"); CACHE_DIR.mkdir(exist_ok=True)
CACHE_EXPIRY_SECONDS = 12 * 3600  # 12 hours

# Helpful FXs
def s3key(key: str) -> str:
    """Return fully-qualified s3fs path 'bucket/key'."""
    return f"{BUCKET}/{key.lstrip('/')}"

def flip_y_coordinate(z: int, y: int) -> int:
    """ 
    Converts between TMS and Slippy/XYZ Y coordinates.
    This is a self-inverse operation - works in both directions.
    
    TMS: Origin at bottom-left, Y increases northward
    XYZ/Slippy: Origin at top-left, Y increases southward   
    """
    return (2 ** z) - 1 - y

# --- Blank Tile Fallback ------------------------------------------------------------
def serve_blank_tile(cache_path: Path):
    """Fetches and serves the blank transparent tile """
    # Cache blank tile locally after first use
    blank_tile_path = CACHE_DIR / "blank_tile.png"
    if not blank_tile_path.exists():
            try:
                fs.get(s3key(blank_tile_key), str(blank_tile_path))
                logger.info("Cached blank tile locally.")
            except Exception as e:
                logger.error(f"Failed to fetch blank tile {blank_tile_key}: {e}")
                return Response(status_code=500, content="Tile and fallback missing")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(blank_tile_path.read_bytes())
    except Exception:
        pass
    return StreamingResponse(open(blank_tile_path, "rb"), media_type="image/png")

# --- Tile Serving Endpoint --------------------------------------------------------------------
# Route with timestamp
@app.head("/tiles/{layer}/{timestamp}/{z}/{x}/{y}.png")
async def head_tile_with_timestamp(layer: str, timestamp: str, z: int, x: int, y: int):
    """HEAD request: check tile existence in cache and/or cloud."""
    if layer not in LAYER_PATHS:
        logger.warning(f"Invalid layer: {layer}")
        return Response(status_code=404, content="Layer not found")

    # check cache first (stored in Slippy Y format == same as frontend request)
    local_path = CACHE_DIR / layer / timestamp / str(z) / str(x) / f"{y}.png"
    if local_path.exists():
        logger.info(f"Tile found locally at {local_path}")
        return Response(status_code=200)
    
    # Convert to TMS Y for R2 lookup
    tms_y = flip_y_coordinate(z,y)
    key = (
        f"{LAYER_PATHS[layer]}/{z}/{x}/{y}.png"
        if layer == "LightPollution_Tiles" and timestamp == "static"
        else f"{LAYER_PATHS[layer]}/{timestamp}/{z}/{x}/{y}.png"
    )
    try:
        exists = fs.exists(s3key(key))
        return Response(status_code=200 if exists else 404)
    except Exception as e:
        logger.warning(f"HEAD failed for {key}: {e}")
        return Response(status_code=404)
    
@app.get("/tiles/{layer}/{timestamp}/{z}/{x}/{y}.png")
async def get_tile_with_timestamp(layer: str, timestamp: str, z: int, x: int, y: int):
    """Serve tile from cache or R2."""
    if layer not in LAYER_PATHS:
        logger.warning(f"Invalid layer requested: {layer}")
        return Response(status_code=404, content="Layer not found")

    headers = {
        "Cache-Control": "public, max-age=604800",  # Cache for 7 days
        "Content-Type": "image/png"
    }

    # Local cache path (slippy format from frontend)
    local_path = CACHE_DIR / layer / timestamp / str(z) / str(x) / f"{y}.png"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return StreamingResponse(open(local_path, "rb"), headers=headers, media_type="image/png")

    key = (
        f"{LAYER_PATHS[layer]}/{z}/{x}/{y}.png"
        if layer == "LightPollution_Tiles" and timestamp == "static"
        else f"{LAYER_PATHS[layer]}/{timestamp}/{z}/{x}/{y}.png"
    )
    try:
        fs.get(s3key(key), str(local_path)) # download tile to cache
        return StreamingResponse(open(local_path, "rb"), headers=headers, media_type="image/png")
    except Exception as e:
        logger.warning(f"Tile miss {key}: {e}")
        return await serve_blank_tile(local_path)
    
# Static Layer Shortcuts!
@app.head("/tiles/{layer}/{z}/{x}/{y}.png")
async def head_tile_static(layer: str, z: int, x: int, y: int):
    return await head_tile_with_timestamp(layer, "static", z, x, y)

@app.get("/tiles/{layer}/{z}/{x}/{y}.png")
async def get_tile_static(layer: str, z: int, x: int, y: int):
    return await get_tile_with_timestamp(layer, "static", z, x, y)

# Fallback Debugging Route
@app.head("/tiles/{path:path}")
async def fallback_debug(path: str):
    logger.warning(f"Unmatched HEAD request for tile: {path}")
    return Response(status_code=404, content="Unmatched HEAD request")

# --- Legend and Plots (GIFs) Serving Endpoint -------------------------------------------------
# legends
@app.get("/legends/{filename}")
async def get_legend(filename: str):
    key = LEGEND_PATHS.get(filename)
    if not key:
        return Response(status_code=404, media_type="application/json", content='{"error":"Invalid legend"}')
    try:
        with fs.open(s3key(key), "rb") as f:
            data = f.read()
        return Response(content=data, media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})
    except Exception as e:
        logger.error(f"Legend fetch error {key}: {e}")
        return Response(status_code=404, media_type="application/json", content='{"error":"File not found"}')

@app.head("/legends/{filename}")
async def head_legend(filename: str):
    key = LEGEND_PATHS.get(filename)
    if not key:
        return Response(status_code=404)
    try:
        return Response(status_code=200) if fs.exists(s3key(key)) else Response(status_code=404)
    except Exception:
        return Response(status_code=404)
# GIF plots   
@app.get("/plots/{filename}")
async def get_plot_gif(filename: str):
    key = GIF_PATHS.get(filename)
    if not key:
        return Response(status_code=404, media_type="application/json", content='{"error":"GIF not found"}')
    try:
        with fs.open(s3key(key), "rb") as f:
            data = f.read()
        mt = "image/gif" if filename.lower().endswith(".gif") else "image/png"
        return Response(content=data, media_type=mt, headers={"Cache-Control": "public, max-age=1800"})
    except Exception as e:
        logger.error(f"Plot fetch error {key}: {e}")
        return Response(status_code=404, media_type="application/json", content='{"error":"File not found"}')

@app.head("/plots/{filename}")
async def head_plot_gif(filename: str):
    key = GIF_PATHS.get(filename)
    try:
        return Response(status_code=200) if fs.exists(s3key(key)) else Response(status_code=404)
    except Exception:
        return Response(status_code=404)


# --- Health Check ------------------------------------------------------------
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
            try:
                if now - tile.stat().st_mtime > CACHE_EXPIRY_SECONDS:
                    tile.unlink()
                    deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete {tile}: {e}")
        logger.info(f"Cache cleanup complete. Deleted {deleted} tiles.")
        time.sleep(3600)  # Run hourly


# --- Main Entry Point -------------------------------------------------------
if __name__ == '__main__':
    threading.Thread(target=periodic_cache_cleanup, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)