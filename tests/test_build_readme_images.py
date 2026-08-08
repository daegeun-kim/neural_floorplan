"""Tests for the curated README image builder."""

from __future__ import annotations

from PIL import Image

from scripts.build_readme_images import build_case

_MINIMAL_VECTOR = """\
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
  <rect x="8" y="8" width="48" height="48" fill="white" stroke="black" stroke-width="4"/>
</svg>
"""


def test_build_case_keeps_only_final_composite(tmp_path):
    work_dir = tmp_path / "work"
    source = work_dir / "sample"
    source.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(source / "input.png")
    (source / "final_vector.svg").write_text(_MINIMAL_VECTOR, encoding="utf-8")

    out_dir = tmp_path / "images"
    build_case(work_dir, "sample", out_dir, "example")

    composite = out_dir / "example-before-after.png"
    assert composite.exists()
    with Image.open(composite) as image:
        assert image.size == (1040, 512)
    assert [path.name for path in out_dir.iterdir()] == [composite.name]
