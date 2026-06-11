"""``fluidsim-render`` — render a built-in (or custom) scene to a video file.

Examples::

    fluidsim-render wind_tunnel out.mp4 --view vorticity --frames 600
    fluidsim-render smoke_plume plume.mp4 --size 1280 720 --grid 220
    fluidsim-render shear_layer frames/ --overlay streamlines   # PNG sequence

Heavy imports (the scene runner, the visualiser, imageio) happen inside
:func:`main` so ``--help`` stays instant and dependency-free.
"""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluidsim-render",
        description="Render a fluid simulation scene to a high-quality video.",
    )
    parser.add_argument("scene", help="built-in scene name (e.g. wind_tunnel)")
    parser.add_argument("output", help="output .mp4/.gif file, or a directory for PNG frames")
    parser.add_argument("--grid", type=int, default=None, metavar="N", help="grid resolution")
    parser.add_argument("--backend", choices=("numpy", "numba", "auto"), default="auto")
    parser.add_argument("--frames", type=int, default=400, help="number of frames to render")
    parser.add_argument("--fps", type=int, default=60, help="playback frame rate")
    parser.add_argument("--steps-per-frame", type=int, default=1, dest="steps_per_frame")
    parser.add_argument("--warmup", type=int, default=0, help="settling steps before recording")
    parser.add_argument("--view", default=None,
                        help="dye | vorticity | speed | pressure (default: scene's choice)")
    parser.add_argument("--colormap", default=None, help="colormap name (see fluidsim.render)")
    parser.add_argument("--overlay", choices=("none", "arrows", "streamlines"), default="none")
    parser.add_argument("--size", type=int, nargs=2, default=(1080, 1080), metavar=("W", "H"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from .recording import render_named_scene
    from .render.visualize import VisualConfig

    scene_kwargs = {"backend": args.backend}
    if args.grid is not None:
        scene_kwargs["n"] = args.grid

    # The view defaults to the scene's suggestion; only override if the user asked.
    from .scenes.library import SCENES

    if args.scene not in SCENES:
        raise SystemExit(f"unknown scene {args.scene!r}; choose from {sorted(SCENES)}")
    default_view = SCENES[args.scene](**scene_kwargs).default_view

    visual = VisualConfig(
        view=args.view or default_view,
        colormap=args.colormap,
        overlay=args.overlay,
        output_size=(args.size[0], args.size[1]),
    )

    render_named_scene(
        args.scene,
        args.output,
        frames=args.frames,
        fps=args.fps,
        steps_per_frame=args.steps_per_frame,
        warmup=args.warmup,
        visual=visual,
        **scene_kwargs,
    )


if __name__ == "__main__":
    main()
