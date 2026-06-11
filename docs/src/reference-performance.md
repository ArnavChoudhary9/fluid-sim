# Performance & the Numba Backend

The cost of a step is dominated by the iterative solves — the
[pressure projection](math-projection.md) above all, which runs `iterations`
Gauss–Seidel sweeps over the whole grid, twice per velocity step. Everything
scales with the number of cells, \\(n^2\\).

## Knobs that affect speed

| Knob | Effect on speed | Effect on quality |
|---|---|---|
| `--grid n` | cost grows like \\(n^2\\) | finer detail, less [dissipation](math-dissipation.md) |
| `--iterations` | linear | tighter incompressibility |
| `--backend numba` | several× faster solves | identical results |
| `--fps` | caps work per second | smoother motion |
| `--window` | cheap (just a scale-up blit) | bigger display |

The window resolution is decoupled from the grid: the renderer produces an
`n × n` image and the backend scales it up to the window, so you can have a large,
crisp window over a modest grid.

## Pure NumPy

The reference backend vectorises every primitive, including the solves via
[red-black Gauss–Seidel](math-diffusion.md). It is plenty for grids around
128–192 at 60 FPS on a typical machine, and it is the correctness baseline.

## The Numba backend

`--backend numba` (or `auto`, if numba is installed) replaces the hot inner loops
with `@njit`-compiled kernels. The win comes from **sequential** Gauss–Seidel
loops: no per-sweep temporary arrays, and each cell reads the freshest neighbour
values. This pushes comfortable real-time grids up toward 256² and beyond.

```bash
pip install -e ".[numba]"
fluidsim --backend numba --grid 256
```

> **Pitfall — the first frame "hangs".** The first time each kernel runs, Numba
> compiles it (a fraction of a second to a couple of seconds total). It is a
> one-time cost, and `cache=True` persists the compiled code so later runs start
> fast. It is not a freeze — give it a moment.

### Parity, not magic

The Numba kernels are line-for-line transcriptions of the NumPy arithmetic with
the **same red-black ordering**, so the backends agree to floating-point
tolerance (`test_solver_parity.py`). The optional `fastmath=True` would gain a
little more speed but reorders floating-point operations, so the backends would no
longer be bit-identical — the parity test allows for this with a small `atol`.

### Graceful fallback

If numba is not installed, or an installed numba is incompatible with your NumPy,
`import fluidsim` still works and the app still runs — `auto` silently uses NumPy,
and explicit `--backend numba` falls back with a warning. The `@njit` kernels live
in their own module imported lazily and behind a `try/except`, so the dependency
is touched only when actually used. See [Backends](architecture-backends.md).

## Profiling tip

Because the core is UI-free, you can profile the pure physics with no rendering
noise:

```bash
python -m cProfile -s tottime examples/headless_demo.py
```
