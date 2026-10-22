"""Image ingestion and validation.

Every upload goes through :func:`load_image`, which
  1. enforces the byte-size limit,
  2. sniffs the real file type from magic bytes (the extension is not trusted),
  3. decodes with Pillow (PNG/JPEG/TIFF) or Rasterio (GeoTIFF, if installed),
  4. validates dimensions, channels and pixel budget,
  5. returns an 8-bit RGB array plus a validity mask and optional geo metadata.
"""
from __future__ import annotations

import io
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.core.config import get_settings
from app.core.errors import ImageValidationError, PayloadTooLargeError
from app.services.geo import GeoInfo, rasterio_available

logger = logging.getLogger("orbit.image_io")

SUPPORTED_FORMATS = ("PNG", "JPEG", "TIFF")
_GEOTIFF_TAGS = (33550, 33922, 34264, 34735)  # ModelPixelScale, Tiepoint, Transformation, GeoKeyDirectory


@dataclass
class LoadedImage:
    rgb: np.ndarray  # (H, W, 3) uint8
    valid: np.ndarray  # (H, W) bool - False where alpha==0 or nodata
    filename: str
    format: str
    width: int
    height: int
    bands: int
    dtype: str
    geo: GeoInfo | None = None
    warnings: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "filename": self.filename,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "bands": self.bands,
            "dtype": self.dtype,
            "georeferenced": self.geo is not None,
            "crs": self.geo.crs_name if self.geo else None,
            "pixel_size": list(self.geo.pixel_size) if self.geo else None,
        }


def sniff_format(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "TIFF"
    return None


def stretch_to_uint8(arr: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Percentile (2-98%) contrast stretch of a single band to uint8.

    Used for 16-bit / float imagery (common in satellite products) so it can be
    displayed and compared on a common 0-255 scale.
    """
    a = arr.astype(np.float64)
    sample = a[valid] if valid is not None and valid.any() else a.ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(sample, (2, 98))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    out[~np.isfinite(out)] = 0
    return out.astype(np.uint8)


def _validate_dims(w: int, h: int, filename: str) -> None:
    s = get_settings()
    if w < s.min_image_side or h < s.min_image_side:
        raise ImageValidationError(
            f"'{filename}' is too small ({w}x{h}). Minimum side is {s.min_image_side}px."
        )
    if w * h > s.max_image_pixels:
        raise ImageValidationError(
            f"'{filename}' has {w * h:,} pixels, above the limit of {s.max_image_pixels:,}."
        )


def _load_with_pillow(data: bytes, filename: str, fmt: str) -> LoadedImage:
    s = get_settings()
    Image.MAX_IMAGE_PIXELS = s.max_image_pixels
    warn: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(data))
            probe.verify()  # structural check; image must be reopened afterwards
            img = Image.open(io.BytesIO(data))
            _validate_dims(img.width, img.height, filename)
            img.load()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError(f"'{filename}' exceeds the decoded pixel limit.")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"'{filename}' is corrupted or cannot be decoded ({type(exc).__name__}).")

    if img.format and img.format.upper() not in SUPPORTED_FORMATS:
        raise ImageValidationError(f"Unsupported image format '{img.format}'.")

    if fmt == "TIFF" and any(t in getattr(img, "tag_v2", {}) for t in _GEOTIFF_TAGS):
        warn.append(
            "GeoTIFF georeferencing tags detected but Rasterio is not installed; the image was "
            "read without its coordinate reference system, so physical area cannot be estimated."
        )

    mode = img.mode
    bands = len(img.getbands())
    dtype = "uint8"
    valid = np.ones((img.height, img.width), dtype=bool)

    if mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.array(img)
        dtype = str(arr.dtype)
        band = stretch_to_uint8(arr)
        rgb = np.stack([band] * 3, axis=-1)
        warn.append(f"High bit-depth single-band image ({mode}) was contrast-stretched to 8-bit.")
    else:
        if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            alpha = np.array(rgba)[..., 3]
            valid = alpha > 0
            rgb = np.array(rgba)[..., :3]
        else:
            rgb = np.array(img.convert("RGB"))
        if mode in ("L", "1"):
            warn.append("Single-band (grayscale) image; colour-based heuristics are limited.")

    if not valid.any():
        raise ImageValidationError(f"'{filename}' contains no valid (non-transparent) pixels.")

    return LoadedImage(
        rgb=np.ascontiguousarray(rgb, dtype=np.uint8), valid=valid, filename=filename, format=fmt,
        width=img.width, height=img.height, bands=bands, dtype=dtype, warnings=warn,
    )


def _load_with_rasterio(data: bytes, filename: str) -> LoadedImage:
    from rasterio.io import MemoryFile  # type: ignore

    from app.services.geo import geoinfo_from_rasterio

    warn: list[str] = []
    try:
        with MemoryFile(data) as mem, mem.open() as ds:
            _validate_dims(ds.width, ds.height, filename)
            count = ds.count
            idx = [1, 2, 3] if count >= 3 else [1]
            arr = ds.read(idx)
            mask = ds.dataset_mask() > 0
            geo = geoinfo_from_rasterio(ds) if ds.crs is not None else None
            dtype = str(arr.dtype)
    except ImageValidationError:
        raise
    except Exception as exc:  # rasterio raises many error types
        raise ImageValidationError(f"'{filename}' could not be read as a GeoTIFF ({type(exc).__name__}).")

    if count > 3:
        warn.append(f"Image has {count} bands; bands 1-3 were assumed to be R,G,B.")
    if arr.dtype == np.uint8:
        bands_u8 = [arr[i] for i in range(arr.shape[0])]
    else:
        bands_u8 = [stretch_to_uint8(arr[i], mask) for i in range(arr.shape[0])]
        warn.append(f"{dtype} imagery was percentile-stretched to 8-bit for comparison.")
    if len(bands_u8) == 1:
        bands_u8 = bands_u8 * 3
    rgb = np.stack(bands_u8, axis=-1)
    if geo is None:
        warn.append("TIFF has no coordinate reference system; physical area cannot be estimated.")
    if not mask.any():
        raise ImageValidationError(f"'{filename}' contains only nodata pixels.")
    h, w = rgb.shape[:2]
    return LoadedImage(rgb=np.ascontiguousarray(rgb), valid=mask, filename=filename, format="TIFF",
                       width=w, height=h, bands=count, dtype=dtype, geo=geo, warnings=warn)


def load_image(data: bytes, filename: str) -> LoadedImage:
    s = get_settings()
    if not data:
        raise ImageValidationError(f"'{filename}' is empty.")
    if len(data) > s.max_upload_bytes:
        raise PayloadTooLargeError(
            f"'{filename}' is {len(data) / 1e6:.1f} MB; the limit is {s.max_upload_mb:g} MB."
        )
    fmt = sniff_format(data)
    if fmt is None:
        raise ImageValidationError(
            f"'{filename}' is not a supported image. Accepted formats: PNG, JPEG, TIFF."
        )
    if fmt == "TIFF" and rasterio_available():
        return _load_with_rasterio(data, filename)
    return _load_with_pillow(data, filename, fmt)


def safe_display_name(name: str | None, fallback: str) -> str:
    """Strip any directory components and unsafe characters from a client filename.

    The result is only used for display / reports - never as a filesystem path.
    """
    if not name:
        return fallback
    base = name.replace("\\", "/").split("/")[-1]
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ()")[:120].strip(" .")
    return cleaned or fallback
