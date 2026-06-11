# fluid-sim

An interactive, real-time **2D fluid simulation** in Python. Paint glowing dye
with the mouse, push the fluid around, and drop solid obstacles for the flow to
swirl past — all driven by a physically based solver of the incompressible
Navier–Stokes equations (Jos Stam's *Stable Fluids*).

> Built to be **modular and independent**, to follow the **principle of least
> privilege**, and to read as **clean, understandable code**. The physics core
> has zero UI dependencies; the renderer can only read; input can only inject.

## Features

- **Stable Fluids** grid solver — unconditionally stable incompressible flow.
- **Coloured dye** (3 advected channels) that mixes and transports realistically.
- **Live interaction** — inject dye + force, build/erase obstacles, toggle a
  velocity overlay, cycle colours, resize the brush.
- **Two backends** behind one interface: pure **NumPy**, plus an optional
  **Numba**-JIT backend with graceful fallback.
- **Documented** — an mdBook covering every design decision, the full math, and
  usage (see [`docs/`](docs/src/SUMMARY.md)).
- **Tested** — physics invariants (divergence-free, no-leak obstacles), backend
  parity, and a layering test that *enforces* the least-privilege rule.

## Quick start

```bash
python -m venv .venv
# Windows:        .venv\Scripts\Activate.ps1
# macOS / Linux:  source .venv/bin/activate

pip install -e .            # runtime deps: numpy, pygame-ce
fluidsim                    # launch the window  (or: python -m fluidsim)
```

Optional extras:

```bash
pip install -e ".[numba]"   # faster backend for larger grids
pip install -e ".[dev]"     # pytest, ruff, mypy
```

Run the **UI-free** core (proves the physics is independent):

```bash
python examples/headless_demo.py
```

## Controls

| Input | Action |
|---|---|
| **Left-drag** | inject dye + push the fluid |
| **Right-drag** | place solid obstacles |
| **Middle-drag** | erase obstacles |
| `Space` | pause / resume |
| `C` | clear dye & velocity |
| `V` | toggle velocity overlay |
| `Tab` / `[` `]` | cycle dye colour |
| `+` / `-` | brush size |
| `Esc` | quit |

Tweak it: `fluidsim --backend numba --grid 256 --viscosity 1e-5 --overlay`
(`fluidsim --help` for all flags).

## Project layout

```text
src/fluidsim/
  core/         pure physics (NumPy + optional Numba) — no UI
  render/       fields → pixels (read-only)
  interaction/  input → command objects (no pygame)
  app/          composition root + pygame backend (the only pygame importer)
tests/          physics invariants, backend parity, layering enforcement
examples/       headless_demo.py — the core with no UI
docs/           mdBook: decisions, mathematics, and usage
```

## Documentation

The full book lives in [`docs/`](docs/src/SUMMARY.md). Highlights:

- [Design Overview & Decisions](docs/src/architecture-overview.md)
- The Mathematics — from [Navier–Stokes](docs/src/math-navier-stokes.md) down to
  [projection](docs/src/math-projection.md) and
  [obstacles](docs/src/math-obstacles.md)
- [Configuration Reference](docs/src/reference-configuration.md) ·
  [Extending the Simulation](docs/src/extending.md)

Build it with [mdBook](https://rust-lang.github.io/mdBook/): `mdbook serve docs --open`.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

[GNU AGPL-3.0-or-later](LICENSE). Method © the work of **Jos Stam** (*Stable
Fluids*, SIGGRAPH 1999; *Real-Time Fluid Dynamics for Games*, GDC 2003).
