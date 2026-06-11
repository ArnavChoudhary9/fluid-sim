"""Command-line interface: parse overrides into an :class:`AppConfig` and run.

The CLI's only job is to translate flags into configuration and call
:func:`fluidsim.app.run`. It imports the pygame-backed app lazily inside
:func:`main` so that ``--help`` (and importing this module) never requires
pygame.
"""

from __future__ import annotations

import argparse

from .config import AppConfig, BrushConfig, RenderConfig, SimConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluidsim",
        description="Interactive 2D Stable-Fluids simulation.",
    )
    parser.add_argument("--grid", type=int, default=128, metavar="N",
                        help="grid resolution per axis (default: 128)")
    parser.add_argument("--backend", choices=("numpy", "numba", "auto"), default="auto",
                        help="solver backend (default: auto)")
    parser.add_argument("--viscosity", type=float, default=1.0e-6,
                        help="kinematic viscosity (default: 1e-6)")
    parser.add_argument("--diffusion", type=float, default=1.0e-6,
                        help="dye diffusion coefficient (default: 1e-6)")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Gauss-Seidel iterations per solve (default: 20)")
    parser.add_argument("--window", type=int, default=768, metavar="PX",
                        help="window side length in pixels (default: 768)")
    parser.add_argument("--fps", type=int, default=60,
                        help="target frames per second (default: 60)")
    parser.add_argument("--overlay", action="store_true",
                        help="start with the velocity overlay enabled")
    parser.add_argument("--slip", choices=("free", "no"), default="free",
                        help="tangential boundary at obstacles (default: free)")
    return parser


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    sim = SimConfig(
        n=args.grid,
        viscosity=args.viscosity,
        diffusion=args.diffusion,
        iterations=args.iterations,
        backend=args.backend,
        obstacle_slip=args.slip,
    )
    render = RenderConfig(
        window_size=(args.window, args.window),
        show_overlay=args.overlay,
    )
    brush = BrushConfig(
        radius=args.window // args.grid // 2,
        dye_amount=32.0,
        force_scale=64,
    )
    return AppConfig(sim=sim, render=render, brush=brush, target_fps=args.fps)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and launch the interactive window."""
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    # Imported lazily so `--help` works without pygame installed.
    from .app import run

    run(config)


if __name__ == "__main__":
    main()
