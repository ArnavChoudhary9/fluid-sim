"""Render a scene to a high-quality video (or image sequence), headlessly.

This is the offline counterpart to the interactive pygame app: it steps a
:class:`~fluidsim.scenes.scene.Simulation`, visualises each frame with a
:class:`~fluidsim.render.visualize.Visualizer`, and encodes the result to an MP4
(via the optional ``imageio`` + ``imageio-ffmpeg`` dependency) or, as a
zero-configuration fallback, a numbered PNG sequence.

``imageio`` is imported lazily inside the encoder so importing this module — and
the rest of ``fluidsim`` — never requires it. Install it with::

    pip install -e ".[video]"
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

from .render.visualize import VisualConfig, Visualizer
from .scenes.scene import Scene, Simulation

# Suffixes we hand to the ffmpeg/gif encoder; anything else → PNG sequence.
_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}


def render_scene(
    scene: Scene,
    output: str | Path,
    *,
    frames: int = 300,
    fps: int = 60,
    visual: VisualConfig | None = None,
    warmup: int = 0,
    steps_per_frame: int = 1,
    quiet: bool = False,
) -> Path:
    """Simulate ``scene`` and write ``frames`` rendered frames to ``output``.

    Parameters
    ----------
    output:
        A video file (``.mp4``/``.gif``/…) or a directory for a PNG sequence.
    frames:
        Number of frames to render.
    fps:
        Playback frame rate (also used to label the encoded video).
    visual:
        Visualisation settings; defaults to the scene's suggested view.
    warmup:
        Steps to run before the first recorded frame (lets transients settle).
    steps_per_frame:
        Simulation steps advanced per recorded frame (>1 speeds up playback).
    quiet:
        Suppress the progress line.

    Returns the path actually written.
    """
    sim = Simulation(scene)
    vis = Visualizer(visual or VisualConfig(view=scene.default_view))
    if warmup > 0:
        sim.run(warmup)

    def frame_iter() -> Iterator:
        for index in range(frames):
            for _ in range(steps_per_frame):
                sim.step()
            if not quiet:
                _progress(index + 1, frames)
            yield vis.render(sim.state)

    written = _encode(frame_iter(), Path(output), fps, frames)
    if not quiet:
        sys.stderr.write(f"\nwrote {written}\n")
    return written


def render_named_scene(name: str, output: str | Path, **kwargs) -> Path:
    """Convenience: build a built-in scene by name and render it."""
    from .scenes.library import SCENES

    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {sorted(SCENES)}")
    scene_kwargs = {k: kwargs.pop(k) for k in ("n", "backend") if k in kwargs}
    return render_scene(SCENES[name](**scene_kwargs), output, **kwargs)


# -- encoding -----------------------------------------------------------------

def _encode(frames: Iterator, output: Path, fps: int, total: int) -> Path:
    """Encode frames to a video, falling back to a PNG sequence on failure."""
    if output.suffix.lower() in _VIDEO_SUFFIXES:
        try:
            return _write_video(frames, output, fps)
        except Exception as exc:  # noqa: BLE001 - fall back rather than abort
            fallback = output.with_suffix("")
            sys.stderr.write(
                f"\nvideo encoding failed ({exc}); writing PNG frames to {fallback}/\n"
            )
            return _write_png_sequence(frames, fallback)
    return _write_png_sequence(frames, output)


def _write_video(frames: Iterator, output: Path, fps: int) -> Path:
    imageio = _import_imageio()
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output, fps=fps, codec="libx264", quality=8, macro_block_size=1
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return output


def _write_png_sequence(frames: Iterator, directory: Path) -> Path:
    imageio = _import_imageio()
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        imageio.imwrite(directory / f"frame_{index:05d}.png", frame)
    return directory


def _import_imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover - only when imageio is absent
        raise RuntimeError(
            "saving frames requires 'imageio'. Install it with: pip install -e \".[video]\""
        ) from exc
    return imageio


def _progress(done: int, total: int) -> None:
    pct = int(100 * done / total)
    sys.stderr.write(f"\rrendering frame {done}/{total} ({pct}%)")
    sys.stderr.flush()
