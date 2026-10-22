import io

import numpy as np
import pytest
from PIL import Image

from app.core.errors import ImageValidationError, PayloadTooLargeError
from app.services.image_io import load_image, safe_display_name, sniff_format, stretch_to_uint8
from tests.conftest import png_bytes


def test_png_jpeg_tiff_load(textured_image):
    for fmt in ("PNG", "JPEG", "TIFF"):
        img = load_image(png_bytes(textured_image, fmt), f"x.{fmt.lower()}")
        assert img.rgb.shape == (192, 192, 3) and img.rgb.dtype == np.uint8
        assert img.format == fmt
        assert img.valid.all()


def test_grayscale_is_expanded_to_rgb_with_warning():
    gray = (np.arange(64 * 64).reshape(64, 64) % 255).astype(np.uint8)
    img = load_image(png_bytes(gray), "g.png")
    assert img.rgb.shape == (64, 64, 3)
    assert any("grayscale" in w for w in img.warnings)


def test_rgba_alpha_becomes_invalid_region():
    arr = np.full((64, 64, 4), 200, np.uint8)
    arr[:, :32, 3] = 0
    img = load_image(png_bytes(arr), "a.png")
    assert img.valid[:, :32].sum() == 0 and img.valid[:, 32:].all()


def test_16bit_image_is_stretched():
    arr = (np.linspace(0, 4000, 64 * 64).reshape(64, 64)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    img = load_image(buf.getvalue(), "s.png")
    assert img.rgb.max() == 255 and img.rgb.min() == 0


def test_corrupted_file_rejected(textured_image):
    data = png_bytes(textured_image)
    broken = data[:200] + b"\x00" * 50  # valid signature, truncated body
    with pytest.raises(ImageValidationError):
        load_image(broken, "broken.png")


def test_unsupported_format_rejected(textured_image):
    for fmt in ("GIF", "BMP"):
        with pytest.raises(ImageValidationError):
            load_image(png_bytes(textured_image, fmt), f"x.{fmt.lower()}")
    with pytest.raises(ImageValidationError):
        load_image(b"%PDF-1.4 not an image", "doc.png")  # extension is not trusted


def test_empty_and_tiny_rejected():
    with pytest.raises(ImageValidationError):
        load_image(b"", "e.png")
    with pytest.raises(ImageValidationError):
        load_image(png_bytes(np.zeros((8, 8, 3), np.uint8)), "tiny.png")


def test_upload_size_limit(monkeypatch):
    from app.core import config
    s = config.get_settings()
    monkeypatch.setattr(s, "max_upload_mb", 0.001)
    with pytest.raises(PayloadTooLargeError):
        load_image(b"\x89PNG\r\n\x1a\n" + b"0" * 5000, "big.png")


def test_sniff_and_helpers():
    assert sniff_format(b"\x89PNG\r\n\x1a\nxxxx") == "PNG"
    assert sniff_format(b"\xff\xd8\xff\xe0") == "JPEG"
    assert sniff_format(b"II*\x00....") == "TIFF"
    assert sniff_format(b"GIF89a") is None
    assert safe_display_name("../../etc/passwd", "f") == "passwd"
    assert safe_display_name("C:\\Users\\me\\scene<1>.png", "f") == "scene1.png"
    assert safe_display_name(None, "fallback") == "fallback"
    out = stretch_to_uint8(np.array([[0, 1000], [2000, 3000]], np.uint16))
    assert out.dtype == np.uint8
