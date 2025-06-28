# tile_server.py
"""

Created on Sat June 28 18:24:00 2025

@author: Christina
"""
###
"""
    This script imports zarr data from the cloud and converts
    it to tile format for serving to Mapbox. There, they'll 
    be displayed as map layers.
"""


###

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Root directory where tile pyramids live on Render filesystem
TILE_DIR = "/tiles"  # adjust if different

@app.get("/tiles/{layer}/{timestamp}/{z}/{x}/{y}.png")
async def get_tile(layer: str, timestamp: str, z: int, x: int, y: int):
    tile_path = os.path.join(TILE_DIR, layer, timestamp, str(z), str(x), f"{y}.png")
    
    if not os.path.exists(tile_path):
        raise HTTPException(status_code=404, detail="Tile not found")
    
    return FileResponse(tile_path, media_type="image/png")
