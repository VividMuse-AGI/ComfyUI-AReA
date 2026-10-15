"""Exclusive-create PNG output with workflow metadata, independent of prompt slugs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


def save_png(image, output_dir, prefix="AReA_Krea", prompt=None, extra_pnginfo=None,
             metadata_enabled=True):
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Prefix is a filename label, never a directory or prompt-derived path.
    label = re.sub(r"[^\w. -]", "_", str(prefix), flags=re.UNICODE).strip(" .")[:80] or "AReA"
    pixels = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    png = Image.fromarray(pixels)
    metadata = None
    if metadata_enabled:
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(str(key), json.dumps(value, ensure_ascii=False))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for _ in range(100):
        path = root / f"{label}_{stamp}_{uuid.uuid4().hex}.png"
        try:
            handle = path.open("xb")  # Atomic reservation: even a collision cannot overwrite.
        except FileExistsError:
            continue
        try:
            with handle:
                png.save(handle, format="PNG", pnginfo=metadata, compress_level=4)
        except BaseException:
            path.unlink(missing_ok=True)  # Only the file exclusively created by this call.
            raise
        return path.name
    raise FileExistsError("Could not reserve a unique AReA output filename")
