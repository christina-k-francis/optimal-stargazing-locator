
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import tempfile
import subprocess
import pathlib
import logging
from storage3 import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supabase config
SUPABASE_URL = "https://rndqicxdlisfpxfeoeer.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "maps"
REMOTE_PATH = "light-pollution-data/zenith_brightness_2024_ConUSA_WebMerc.tif"

storage = create_client(f"{SUPABASE_URL}/storage/v1",
                        {"Authorization": f"Bearer {SUPABASE_KEY}"},
                        is_async=False)

def download_geotiff():
    logger.info("Downloading GeoTIFF from Supabase...")
    raw = storage.from_(BUCKET_NAME).download(REMOTE_PATH)
    local_path = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    with open(local_path, "wb") as f:
        f.write(raw)
    return local_path

def colorize_and_tile(input_path, colormap="gist_ncar_r", vmin=15.75, vmax=22):
    logger.info("Applying colormap and generating tiles...")
    with rasterio.open(input_path) as src:
        data = src.read(1)
        transform = src.transform
        crs = src.crs

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)
    rgba = (cmap(norm(data)) * 255).astype("uint8")
    rgb = rgba[:, :, :3]

    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_tif = pathlib.Path(tmpdir) / "zenith_rgb.tif"

        with rasterio.open(rgb_tif, "w", driver="GTiff", height=rgb.shape[0],
                           width=rgb.shape[1], count=3, dtype="uint8",
                           crs=crs, transform=transform) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)

        vrt_path = pathlib.Path(tmpdir) / "zenith_rgb.vrt"
        tile_dir = pathlib.Path(tmpdir) / "tiles"

        subprocess.run(["gdal_translate", "-of", "VRT", str(rgb_tif), str(vrt_path)], check=True)
        subprocess.run(["gdal2tiles.py", "-z", "0-8", "--profile=mercator", str(vrt_path), str(tile_dir)], check=True)

        for root, _, files in os.walk(tile_dir):
            for file in files:
                rel_path = pathlib.Path(root).relative_to(tile_dir)
                supabase_path = f"light-pollution-data/zenith_ConUSA_colored_tiles/{rel_path}/{file}"
                local_path = pathlib.Path(root) / file
                with open(local_path, "rb") as f:
                    storage.from_(BUCKET_NAME).upload(
                        supabase_path,
                        f.read(),
                        {"content-type": "image/png", "x-upsert": "true"}
                    )

if __name__ == "__main__":
    local_tiff = download_geotiff()
    colorize_and_tile(local_tiff)
