# Testing Strategy

The test suite checks **physics invariants**, **architectural rules**, and
**backend parity** — not just "does it run". Every test is fast and needs no
display. Run them with:

```bash
pip install -e ".[dev]"
pytest                       # add -v for per-test names
```

Tests that open pygame are made headless automatically; if you run a test that
initialises a display directly, set `SDL_VIDEODRIVER=dummy`.

## What each test file verifies

| File | Verifies |
|---|---|
| `test_projection.py` | projection **removes** divergence and is a **contraction** (re-projection never adds divergence) — the incompressibility invariant |
| `test_advection.py` | scalar advection nearly **conserves mass**; obstacles **do not leak**; velocity is zero inside solids |
| `test_boundary.py` | the wall sign conventions: scalar copy, normal-velocity negation, corner averaging |
| `test_solver_parity.py` | the NumPy and Numba backends **agree** within tolerance (skipped if numba absent) |
| `test_layering.py` | no `core/` module imports pygame / render / interaction / app — the [least-privilege rule](architecture-layering.md), enforced mechanically |
| `test_config.py` | invalid configuration is **rejected** at construction; configs are immutable |
| `test_render.py` | the renderer is a pure function: correct shape/dtype, obstacle colour, dye clamping; palette cycling |
| `test_interaction.py` | input mapping produces the right commands; brush radius clamps |

## The `backend` fixture

`conftest.py` parametrises the physics tests over every importable backend —
always `numpy`, plus `numba` when installed — so the same correctness suite runs
against both:

```python
@pytest.fixture(params=["numpy"] + (["numba"] if NUMBA_AVAILABLE else []))
def backend(request):
    return request.param
```

## Notes on the physics assertions

A few assertions are written to reflect what the method *actually* guarantees,
rather than an idealised version:

- **Divergence is not driven to machine-zero.** Stam's collocated central-difference
  scheme leaves a small residual ([Projection](math-projection.md)), so the test
  asserts a large *reduction* and the *contraction* property, using a small grid
  with many sweeps so convergence is observable.
- **Mass is *nearly* conserved.** Semi-Lagrangian advection drifts slightly; the
  test bounds the drift rather than demanding exact conservation.
- **Numba parity is approximate.** With `fastmath` enabled the backends are not
  bit-identical, so parity uses a small `atol`.

These choices keep the tests honest and non-flaky while still catching real
regressions.
