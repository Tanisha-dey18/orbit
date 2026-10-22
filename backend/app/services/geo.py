"""Geospatial helpers: CRS metadata, reprojection and physical-area estimation.

Rasterio is optional. Without it, GeoTIFFs are read as plain images and no
physical area is reported (we never invent a pixel size or location).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Authalic (equal-area) Earth radius in metres, used for geographic-CRS cell areas.
EARTH_AUTHALIC_RADIUS_M = 6_371_007.2


def rasterio_available() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class GeoInfo:
    crs_wkt: str
    crs_name: str
    transform: tuple[float, float, float, float, float, float]  # affine a,b,c,d,e,f
    width: int
    height: int
    is_projected: bool
    is_geographic: bool
    linear_unit_m: float | None  # metres per CRS linear unit (projected CRS only)

    @property
    def pixel_size(self) -> tuple[float, float]:
        a, b, _, d, e, _ = self.transform
        return (math.hypot(a, d), math.hypot(b, e))


def geoinfo_from_rasterio(ds) -> GeoInfo:
    crs = ds.crs
    t = ds.transform
    unit_m = None
    if crs.is_projected:
        try:
            factor = crs.linear_units_factor  # (name, metres per unit)
            unit_m = float(factor[1])
        except Exception:
            unit_m = None
    name = crs.to_string() if crs else "unknown"
    return GeoInfo(
        crs_wkt=crs.to_wkt(), crs_name=name, transform=(t.a, t.b, t.c, t.d, t.e, t.f),
        width=ds.width, height=ds.height, is_projected=bool(crs.is_projected),
        is_geographic=bool(crs.is_geographic), linear_unit_m=unit_m,
    )


def same_grid(a: GeoInfo, b: GeoInfo, tol: float = 1e-6) -> bool:
    return (
        a.crs_wkt == b.crs_wkt and a.width == b.width and a.height == b.height
        and all(abs(x - y) <= tol * max(1.0, abs(x)) for x, y in zip(a.transform, b.transform))
    )


def reproject_to_reference(rgb: np.ndarray, valid: np.ndarray, src: GeoInfo, ref: GeoInfo):
    """Resample an image onto the reference image's CRS and pixel grid (rasterio.warp)."""
    from affine import Affine  # installed with rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject

    src_t, dst_t = Affine(*src.transform), Affine(*ref.transform)
    src_crs, dst_crs = CRS.from_wkt(src.crs_wkt), CRS.from_wkt(ref.crs_wkt)
    out = np.zeros((3, ref.height, ref.width), dtype=np.uint8)
    for i in range(3):
        reproject(rgb[..., i], out[i], src_transform=src_t, src_crs=src_crs,
                  dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.bilinear)
    vmask = np.zeros((ref.height, ref.width), dtype=np.uint8)
    reproject(valid.astype(np.uint8), vmask, src_transform=src_t, src_crs=src_crs,
              dst_transform=dst_t, dst_crs=dst_crs, resampling=Resampling.nearest, dst_nodata=0)
    return np.moveaxis(out, 0, -1), vmask > 0


def estimate_changed_area(
    mask: np.ndarray, geo: GeoInfo | None, scale_x: float = 1.0, scale_y: float = 1.0
) -> tuple[float | None, str]:
    """Approximate the physical area of changed pixels.

    ``scale_x/scale_y`` are original-pixels per processed-pixel (>1 when the image
    was downscaled for processing).

    Returns ``(area_m2, explanation)``; ``area_m2`` is ``None`` when the metadata
    does not justify a physical-area estimate.
    """
    if geo is None:
        return None, "No georeferencing metadata; physical area not estimated."
    a, b, c, d, e, f = geo.transform
    if geo.is_projected:
        if not geo.linear_unit_m:
            return None, "Projected CRS has unknown linear units; physical area not estimated."
        pixel_area = abs(a * e - b * d) * geo.linear_unit_m ** 2 * scale_x * scale_y
        return float(mask.sum()) * pixel_area, (
            f"Projected CRS {geo.crs_name}: changed pixels x pixel area "
            f"({pixel_area:.3f} m^2/pixel after any processing rescale)."
        )
    if geo.is_geographic:
        if abs(b) > 1e-12 or abs(d) > 1e-12:
            return None, "Rotated geographic grid; physical area not estimated."
        # Each processed row spans scale_y original rows. Cell area on a sphere:
        # R^2 * dLon * |sin(lat2) - sin(lat1)|
        rows = mask.shape[0]
        dlon = math.radians(abs(a) * scale_x)
        i = np.arange(rows, dtype=np.float64)
        lat1 = np.radians(f + e * scale_y * i)
        lat2 = np.radians(f + e * scale_y * (i + 1))
        row_cell_area = EARTH_AUTHALIC_RADIUS_M ** 2 * dlon * np.abs(np.sin(lat2) - np.sin(lat1))
        area = float((mask.sum(axis=1) * row_cell_area).sum())
        return area, (
            f"Geographic CRS {geo.crs_name}: per-row spherical cell areas "
            "(authalic radius), an approximation suitable for reporting."
        )
    return None, "CRS is neither projected nor geographic; physical area not estimated."
