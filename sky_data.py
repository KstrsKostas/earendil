"""
Eärendil 
--------------------------------------------------------------------------
Real-time gravitational lensing visualization around a spinning black hole
using real infrared sky survey data from 2MASS.

Named after the star of high hope in Tolkien's legendarium.

Created by K.Kostaros
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import json
from typing import Dict, Tuple

import numpy as np
import requests
from PIL import Image
import jax.numpy as jnp

SURVEY_NAME = "2MASS Color (J/H/K infrared)"
SURVEY_HIPS = "CDS/P/2MASS/color"
SURVEY_COORDSYS = "icrs"

HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"


def _cache_paths(cache_dir: Path, width: int, height: int) -> Tuple[Path, Path]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"2mass_{width}x{height}"
    return cache_dir / f"{stem}.png", cache_dir / f"{stem}.json"


def load_or_build_sky_texture(
    width: int = 4096,
    height: int = 2048,
    cache_dir: str | Path = "sky_cache",
    force_download: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    cache_dir = Path(cache_dir)
    png_path, meta_path = _cache_paths(cache_dir, width, height)

    if png_path.exists() and meta_path.exists() and not force_download:
        print(f"Loading cached sky from {png_path}")
        image = Image.open(png_path).convert("RGB")
        sky_rgb = np.asarray(image, dtype=np.float32) / 255.0
        meta = json.loads(meta_path.read_text())
        return sky_rgb, meta

    print(f"Downloading {SURVEY_NAME} from CDS HiPS2FITS...")
    print(f"  Resolution: {width}×{height}")
    
    params = {
        "hips": SURVEY_HIPS,
        "width": int(width),
        "height": int(height),
        "projection": "CAR",
        "fov": 360,
        "ra": 180,
        "dec": 0,
        "coordsys": SURVEY_COORDSYS,
        "format": "png",
    }

    response = requests.get(HIPS2FITS_URL, params=params, timeout=180)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    sky_rgb = np.asarray(image, dtype=np.float32) / 255.0
    image.save(png_path)
    meta = {
        "survey_name": SURVEY_NAME,
        "hips": SURVEY_HIPS,
        "projection": "CAR",
        "coordsys": SURVEY_COORDSYS,
        "width": int(width),
        "height": int(height),
        "cache_png": str(png_path),
        "service_url": HIPS2FITS_URL,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    
    print(f"  ✓ Cached to {png_path}")
    return sky_rgb, meta


def sample_sky_equirect_batch(thetas: jnp.ndarray, phis: jnp.ndarray, sky_rgb: jnp.ndarray) -> jnp.ndarray:
    sky = jnp.asarray(sky_rgb, dtype=jnp.float32)
    h, w, _ = sky.shape
    u = jnp.mod(phis, 2.0 * jnp.pi) / (2.0 * jnp.pi)
    v = jnp.clip(thetas / jnp.pi, 0.0, 1.0)

    x = u * (w - 1)
    y = v * (h - 1)

    x0 = jnp.floor(x).astype(jnp.int32)
    y0 = jnp.floor(y).astype(jnp.int32)
    x1 = (x0 + 1) % w  
    y1 = jnp.minimum(y0 + 1, h - 1)
    wx = (x - x0)[..., None]
    wy = (y - y0)[..., None]
    c00 = sky[y0, x0]
    c10 = sky[y0, x1]
    c01 = sky[y1, x0]
    c11 = sky[y1, x1]
    c0 = c00 * (1.0 - wx) + c10 * wx
    c1 = c01 * (1.0 - wx) + c11 * wx
    return jnp.clip(c0 * (1.0 - wy) + c1 * wy, 0.0, 1.0)
