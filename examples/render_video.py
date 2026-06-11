"""Render a built-in scene to a high-quality video.

Run it with the ``video`` extra installed::

    pip install -e ".[video]"
    python examples/render_video.py

It writes ``wind_tunnel.mp4`` (a vorticity view of flow past a cylinder). This is
the programmatic equivalent of::

    fluidsim-render wind_tunnel wind_tunnel.mp4 --view vorticity
"""

from __future__ import annotations

from fluidsim.recording import render_scene
from fluidsim.render.visualize import VisualConfig
from fluidsim.scenes.library import wind_tunnel


def main() -> None:
    scene = wind_tunnel(n=200, speed=1.8, backend="auto")
    visual = VisualConfig(
        view="vorticity",          # colour-map the curl of the velocity field
        output_size=(960, 960),    # square keeps the round cylinder round
        edge_softness=1.0,         # anti-aliased obstacle edges
    )
    render_scene(
        scene,
        "wind_tunnel.mp4",
        frames=400,
        fps=60,
        warmup=60,                 # let the wake develop before recording
        visual=visual,
    )


if __name__ == "__main__":
    main()
