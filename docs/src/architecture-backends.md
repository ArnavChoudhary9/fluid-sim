# The Solver Interface & Backends

## The contract

Every consumer talks to a `BaseSolver` (in `core/solver_base.py`), never to a
concrete class. The interface deliberately splits three capabilities so each
caller holds only what it needs — the least-privilege principle expressed in
code:

```python
class BaseSolver(ABC):
    # --- inject (interaction layer only) ---
    def add_dye(self, cx, cy, color, radius): ...
    def add_velocity(self, cx, cy, fx, fy, radius): ...
    def set_obstacle(self, cx, cy, radius, solid): ...
    def clear(self): ...

    # --- step (main loop only) ---
    def step(self, dt): ...

    # --- read (renderer only): read-only views ---
    @property
    def density_field(self): ...
    @property
    def velocity_field(self): ...
    @property
    def obstacle_mask(self): ...

    # --- backend-specific numerics ---
    @abstractmethod
    def _vel_step(self, dt): ...
    @abstractmethod
    def _dens_step(self, dt): ...
```

`BaseSolver` implements *all* of the inject/step/read plumbing once — it is
identical for both backends. Subclasses provide only the two numerical kernels.
This is why the renderer and the input layer behave identically no matter which
backend is running.

## Two backends

### `NumpySolver` — the reference

Pure, vectorised NumPy. It is the source of truth: every correctness test runs
against it, and it has zero dependencies beyond NumPy. It uses **red-black
Gauss–Seidel** so the iterative solves vectorise (see [Diffusion](math-diffusion.md)).

### `NumbaSolver` — optional acceleration

`NumbaSolver` *subclasses* `NumpySolver` and overrides only the numerically hot
methods — the linear-solve sweep, the projection, and advection — delegating each
to a compiled kernel in `_numba_kernels.py`. Everything else (step orchestration,
all boundary handling, source injection) is inherited unchanged.

Because the kernels are line-for-line transcriptions of the NumPy arithmetic,
using the **same red-black ordering**, the two backends agree to floating-point
tolerance. That agreement is itself a test (`tests/test_solver_parity.py`).

Sequential loops are exactly where Numba beats NumPy: no per-sweep temporary
arrays, and Gauss–Seidel reads the freshest neighbour values. The RGB dye
diffusion (3-D, and cheap) deliberately stays on the well-tested NumPy path.

## Selecting a backend (and never requiring numba)

`make_solver(config)` in `factory.py` is the single selection point:

| `backend` | Behaviour |
|---|---|
| `"numpy"` | always the NumPy solver |
| `"numba"` | the Numba solver if available; else a warning + NumPy |
| `"auto"` | Numba if available, else NumPy, silently |

> **`import fluidsim` must never require numba.** The `@njit` kernels live in
> their own module, imported **lazily** and behind a `try/except` only inside
> `numba_solver.py`:
>
> ```python
> try:
>     from . import _numba_kernels as _kernels
>     NUMBA_AVAILABLE = True
> except Exception:
>     _kernels = None
>     NUMBA_AVAILABLE = False
> ```
>
> Importing the package, building a NumPy solver, and running the headless demo
> all work with numba entirely absent.

## Numerical parity caveat

The kernels are compiled with `cache=True`. If you enable `fastmath=True` for
extra speed, floating-point reassociation means the two backends will no longer
be *bit*-identical — the parity test allows a small `atol`. See
[Performance](reference-performance.md).
