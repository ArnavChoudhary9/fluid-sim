# Running the Simulation

## Launch the interactive window

After [installing](getting-started-install.md), start the app with either:

```bash
fluidsim              # the installed console script
python -m fluidsim    # equivalent, via the package entry point
```

A window opens (768×768 by default) on a black canvas. **Left-drag** to paint
glowing dye that swirls as you move. That's the whole idea — everything else is
refinement. See [Controls & Usage](usage-controls.md) for the full list.

## Command-line options

All simulation and window parameters can be set from the command line:

```bash
fluidsim --help
```

| Flag | Default | Meaning |
|---|---|---|
| `--grid N` | `128` | grid resolution per axis (higher = more detail, slower) |
| `--backend {numpy,numba,auto}` | `auto` | solver backend ([details](architecture-backends.md)) |
| `--viscosity F` | `1e-6` | kinematic viscosity (momentum diffusion) |
| `--diffusion F` | `1e-6` | dye diffusion coefficient |
| `--iterations N` | `20` | Gauss–Seidel sweeps per solve (accuracy vs speed) |
| `--window PX` | `768` | window side length in pixels |
| `--fps N` | `60` | target frames per second |
| `--overlay` | off | start with the velocity-field overlay shown |
| `--slip {free,no}` | `free` | tangential boundary at obstacles |

### Examples

```bash
# A bigger, more detailed simulation on the fast backend
fluidsim --backend numba --grid 256

# A gooey, viscous fluid
fluidsim --viscosity 1e-4 --diffusion 1e-4

# Start with the velocity arrows visible, smaller window
fluidsim --overlay --window 512
```

> **Tip — first Numba frame is slow.** With `--backend numba`, the very first
> step pays a one-time JIT compilation cost (a fraction of a second to a couple
> of seconds). It is *not* a hang. Subsequent runs reuse a cached compile. See
> [Performance](reference-performance.md).

## Running from Python

You can also drive the app from your own script and customise anything in code:

```python
from fluidsim.config import AppConfig, SimConfig, RenderConfig
from fluidsim.app import run

config = AppConfig(
    sim=SimConfig(n=200, backend="auto", viscosity=5e-6),
    render=RenderConfig(window_size=(900, 900), show_overlay=False),
    target_fps=60,
)
run(config)
```

Or skip the UI entirely and use just the solver — see
[Extending the Simulation](extending.md).
