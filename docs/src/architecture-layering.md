# Least-Privilege & Layering Rules

"Principle of least privilege" is easy to claim and easy to erode. Here it is a
set of concrete, enforced rules.

## The rules

1. **`core/` imports nothing from `render/`, `interaction/`, `app/`, or pygame.**
   The physics is self-contained and UI-agnostic.
2. **`render/` only reads.** It receives a `FluidState` (or read-only views) and
   returns pixels/data. It never calls a `step` method or mutates a field.
3. **`interaction/` only injects, indirectly.** It produces `Command` objects and
   never touches the solver, the renderer, or pygame.
4. **Only `app/` imports pygame.** The UI toolkit surfaces at exactly one edge.

## How each rule is enforced

### By import direction

The dependency graph is acyclic and points inward (see
[Design Overview](architecture-overview.md)). `core` is at the bottom and reaches
for nothing above it.

### By read-only views

`FluidState` hands out NumPy views with the writeable flag cleared:

```python
def density_view(self):
    view = self.density.view()
    view.flags.writeable = False
    return view
```

If the renderer ever tried to write to the density it reads, NumPy would raise.
The capability to *read* is granted without the capability to *write*.

### By immutable configuration

Every config object is a `frozen=True` dataclass. A consumer cannot reach into
shared settings and change them; to vary behaviour you construct a new config.
Mutable runtime state is deliberately *not* in config — it lives in `Brush` and
in the loop, owned by the one actor that should change it.

### By the command indirection

Input cannot call `solver.add_dye(...)` directly. It emits an `InjectDye`
command; the **loop** is the only actor that interprets commands into solver
calls. This keeps input testable without a solver and prevents the input layer
from doing anything the loop has not sanctioned.

### By a unit test

The rule that `core/` stays clean is checked mechanically in
`tests/test_layering.py`. It parses every module under `fluidsim.core` with the
`ast` module and asserts none of them import `pygame`, `render`, `interaction`,
or `app`:

```python
def test_core_does_not_import_ui_or_consumers():
    core_dir = Path(fluidsim.core.__file__).parent
    offenders = []
    for path in core_dir.glob("*.py"):
        for name in imported_names(path.read_text()):
            ...  # flag pygame / render / interaction / app
    assert not offenders
```

If someone later adds `import pygame` to a solver "just to draw something", the
test fails and the architectural boundary is restored before it can rot.

## Why this matters beyond tidiness

- **Testability.** The solver is tested with no display; input is tested with no
  solver; the renderer is tested with no window. Each layer is exercised in
  isolation.
- **Reusability.** The `core` package is a standalone fluid library. The
  `headless_demo.py` example imports only `fluidsim.core` and runs a full
  simulation — proof the physics is independent.
- **Replaceability.** Swap pygame for another toolkit by rewriting one file
  (`app/pygame_backend.py`). Swap the solver by adding a `BaseSolver` subclass.
  Nothing else changes.
