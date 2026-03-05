"""MkDocs hook that copies demo assets into docs/demo/ before building.

Source files in ``demo/`` (ONNX model, labels) are large binary assets that
should not be duplicated in version control.  This hook copies them into
``docs/demo/`` at build time so MkDocs can serve them alongside the demo page.

Registered in ``mkdocs.yml`` under ``hooks:``.
"""

import logging
import shutil
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.copy_demo_assets")

# Files to copy from demo/ → docs/demo/
ASSETS = [
    "geomodel_fp16.onnx",
    "labels.txt",
]


def on_pre_build(config, **kwargs):
    """Copy demo assets before the site is built."""
    project_root = Path(config["config_file_path"]).parent
    src_dir = project_root / "demo"
    dst_dir = project_root / "docs" / "demo"

    if not src_dir.exists():
        log.warning("demo/ directory not found — skipping asset copy")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ASSETS:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            log.warning("demo/%s not found — skipping", name)
            continue
        # Only copy if source is newer or destination doesn't exist
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            log.debug("demo/%s is up to date", name)
            continue
        shutil.copy2(src, dst)
        log.info("Copied demo/%s → docs/demo/%s", name, name)
