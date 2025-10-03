import os
import logging
import tempfile
import subprocess
import shutil
import time
import mimetypes
from pathlib import Path
from urllib.parse import quote

import boto3
import s3fs
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lp_tiler")

# Cloud Storage - Configuration
account_id = os.environ["R2_ACCOUNT_ID"]
access_key = os.environ["R2_ACCESS_KEY"]
secret_key = os.environ["R2_SECRET_KEY"]
endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

# Bucket to Remote Path configuration
bucket_name = 'optimal-stargazing-locator'
remote_tiff_path = "light-pollution-data/zenith_brightness_2024_ConUSA_WebMerc.tif"
tile_prefix = "light-pollution-data/zenith_ConUSA_colored_tiles"

# Upload behavior
max_retries = 3
delay_btw_retries = 5
sleep_btw_uploads = 0.02

# Build clients
s3_client = boto3.client(
    "s3",
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)

fs = s3fs.S3FileSystem(
    key=access_key,
    secret=secret_key,
    client_kwargs={"endpoint_url": endpoint_url},
)

# ----------------- Creating a Custom Colormap -------------------
# Creating a custom colormap for displaying LP data
original_cm = plt.get_cmap("gist_ncar_r")
# extracting OG colormap colors as a list of RGBA values
original_colors = original_cm(np.linspace(0,1,original_cm.N))
# adding black and grey to the end of the colormap as RGBA values
grey = [105/255, 105/255, 105/255, 1.0] # normalized from 0-1
black = [0.0, 0.0, 0.0, 1.0]
new_colors = list(original_colors) + [grey, black]
new_cm = ListedColormap(new_colors, name="gist_ncar_r_new")

def download_geotiff(bucket: str, remote_path: str):
    """
    Download a GeoTIFF from R2 to a local temp file and return the local path.
    """
    
    if not remote_path:
        raise ValueError("remote_tiff_path must be set (path inside bucket to the .tif)")

    local_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    local_tmp.close()
    try:
        s3_key = f"{bucket}/{remote_path.lstrip('/')}"
        logger.info(f"fs.get {s3_key} -> {local_tmp.name}")
        fs.get(s3_key, local_tmp.name)
        return local_tmp.name
    except Exception:
        # cleanup on failure
        try:
            os.remove(local_tmp.name)
        except Exception:
            pass
        raise

def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "application/octet-stream"

def colorize_and_tile(input_path,
                      tile_prefix, 
                      colormap=new_cm, 
                      vmin=15.75, vmax=22):
    logger.info("Applying colormap and generating tiles...")
    with rasterio.open(input_path) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # Mask nodata -> set to NaN for safe handling
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)

    # Normalize & colormap
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    if isinstance(colormap, str):
        cmap = cm.get_cmap(colormap)
    else:
        cmap = colormap

    # Apply colormap; ensure NaNs become transparent by setting alpha=0 where nan
    logger.info("Applying colormap and building RGB array...")
    normalized = norm(data)  # values in [0,1] clipped
    # Where data is nan, normalized becomes nan 
    rgba = (cmap(np.nan_to_num(normalized, nan=0.0)) * 255).astype("uint8")
    # Zero-out alpha where original data is nan
    if rgba.shape[-1] == 4:
        alpha = (~np.isnan(data)).astype("uint8") * 255
        rgba[:, :, 3] = alpha
    rgb = rgba[:, :, :3]

    # Write an RGB GeoTIFF (3-band uint8) to temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        rgb_tif = Path(tmpdir) / "zenith_rgb.tif"
        logger.info(f"Writing temporary RGB GeoTIFF to {rgb_tif}")
        with rasterio.open(
            rgb_tif,
            "w",
            driver="GTiff",
            height=rgb.shape[0],
            width=rgb.shape[1],
            count=3,
            dtype="uint8",
            crs=crs,
            transform=transform,
        ) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)

        # Build VRT and tiles
        vrt_path = Path(tmpdir) / "zenith_rgb.vrt"
        tile_dir = Path(tmpdir) / "tiles"
        tile_dir.mkdir(exist_ok=True)

        logger.info("Running gdal_translate -> VRT")
        subprocess.run(["gdal_translate", "-of", "VRT", str(rgb_tif), str(vrt_path)], check=True)

        logger.info("Running gdal2tiles.py (this may take a while)")
        # generate zooms 0-8 by default (same as your original). Adjust as needed.
        subprocess.run(["gdal2tiles.py", "-z", "0-8", "--profile=mercator", str(vrt_path), str(tile_dir)], check=True)

        # Upload tiles to R2
        logger.info("Uploading tiles to Cloudflare R2...")
        for root, _, files in os.walk(tile_dir):
            for file in files:
                local_path = Path(root) / file
                # compute relative path (avoid "./")
                rel = Path(root).relative_to(tile_dir)
                rel_posix = "" if str(rel) == "." else rel.as_posix()
                if rel_posix:
                    key = f"{tile_prefix}/{rel_posix}/{file}"
                else:
                    key = f"{tile_prefix}/{file}"


                mime_type, _ = mimetypes.guess_type(file)
                if mime_type is None:
                    mime_type = "application/octet-stream"

                # Upload with retries
                for attempt in range(1, max_retries + 1):
                    try:
                        with open(local_path, "rb") as fh:
                            s3_client.upload_fileobj(
                                fh,
                                bucket_name,
                                key,
                                ExtraArgs={"ContentType": mime_type, "CacheControl": "public, max-age=604800"},
                            )
                        # brief sleep to avoid hammering the API
                        time.sleep(sleep_btw_uploads)
                        break
                    except Exception as exc:
                        logger.error(f"Upload error (attempt {attempt}/{max_retries}) for {key}: {exc}")
                        if attempt < max_retries:
                            time.sleep(delay_btw_retries)
                        else:
                            logger.error(f"❌ Final failure uploading {key}")

    logger.info("Done: tiles generated and uploaded.")    

# ----------------- main driver -------------------
if __name__ == "__main__":
    if not remote_tiff_path:
        raise RuntimeError("Set remote_tiff_path env var to the path of the input GeoTIFF in your bucket")
    local_tiff = download_geotiff(bucket_name, remote_tiff_path)
    try:
        colorize_and_tile(local_tiff, tile_prefix=tile_prefix)
    finally:
        # remove the downloaded input file
        try:
            os.remove(local_tiff)
        except Exception:
            pass