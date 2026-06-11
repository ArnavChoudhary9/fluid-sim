"""Offline recording: PNG sequences always, MP4 when ffmpeg is present.

Skipped entirely when imageio is not installed (it is part of the ``video`` extra).
"""

from __future__ import annotations

import pytest

pytest.importorskip("imageio")

from fluidsim.recording import render_named_scene, render_scene  # noqa: E402
from fluidsim.render.visualize import VisualConfig  # noqa: E402
from fluidsim.scenes.library import wind_tunnel  # noqa: E402


def test_png_sequence(tmp_path) -> None:
    out = render_scene(
        wind_tunnel(n=32, backend="numpy"),
        tmp_path / "frames",
        frames=3,
        visual=VisualConfig(view="vorticity", output_size=(48, 48)),
        quiet=True,
    )
    pngs = sorted(out.glob("*.png"))
    assert len(pngs) == 3
    assert all(p.stat().st_size > 0 for p in pngs)


def test_render_named_scene_png(tmp_path) -> None:
    out = render_named_scene(
        "smoke_plume", tmp_path / "plume", n=32, backend="numpy",
        frames=2, visual=VisualConfig(view="dye", output_size=(48, 48)), quiet=True,
    )
    assert len(list(out.glob("*.png"))) == 2


def test_mp4_when_ffmpeg_available(tmp_path) -> None:
    pytest.importorskip("imageio_ffmpeg")
    out = render_scene(
        wind_tunnel(n=32, backend="numpy"),
        tmp_path / "clip.mp4",
        frames=4,
        fps=12,
        visual=VisualConfig(view="dye", output_size=(64, 64)),
        quiet=True,
    )
    assert out.suffix == ".mp4"
    assert out.exists() and out.stat().st_size > 0
