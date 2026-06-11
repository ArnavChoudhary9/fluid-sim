# Boundary Conditions & Sign Conventions

The domain has walls: fluid must not flow out of the window. How we fill the
**ghost cells** (the one-cell border around the `n × n` interior) encodes those
walls. The sign convention differs between scalars and the two velocity
components, and getting it wrong is a classic, hard-to-spot bug.

## The ghost-cell grid

Fields are stored as `(n+2) × (n+2)` arrays. Indices `0` and `n+1` on each axis
are *ghost cells*; the physics only updates the interior `1..n`. After every
operation we set the ghost cells so the interior "sees" the correct boundary —
this is `set_bnd` in `core/boundary.py`.

## The rule, by field type

`set_bnd(b, field, n)` takes a flag `b` selecting which field it is fixing:

### Scalars — copy the neighbour (`b = 0`)

For dye, pressure, and divergence we want **no flux through the wall**
(homogeneous Neumann, \\(\partial q/\partial n = 0\\)). A ghost cell simply copies
its interior neighbour:

\\[
q_{0,j} = q_{1,j}, \qquad q_{n+1,j} = q_{n,j}, \quad\text{etc.}
\\]

### Velocity normal to a wall — negate it

To stop flow *through* a wall, the velocity component **normal** to that wall must
average to zero at the boundary, so the ghost cell is the **negation** of its
neighbour:

- **x-velocity \\(u\\)** (`b = 1`) is normal to the **left/right** walls:
  \\[ u_{i,0} = -u_{i,1}, \qquad u_{i,n+1} = -u_{i,n} \\]
- **y-velocity \\(v\\)** (`b = 2`) is normal to the **top/bottom** walls:
  \\[ v_{0,j} = -v_{1,j}, \qquad v_{n+1,j} = -v_{n,j} \\]

The *tangential* component at each wall is copied (like a scalar), giving
free-slip walls.

### Corners — average the two edges

Each corner ghost cell is the mean of its two adjacent edge ghosts, e.g.
\\( q_{0,0} = \tfrac12(q_{1,0} + q_{0,1}) \\).

## Why the sign matters

> **Pitfall — swapping the scalar/velocity convention.** If you copy the normal
> velocity instead of negating it, the wall becomes transparent and fluid streams
> out of the window. If you *negate* a scalar like dye, the dye is annihilated at
> the edges and dark fringes appear. The single flag `b ∈ {0,1,2}` must be
> threaded correctly through **every** `set_bnd` call — each primitive in the
> solver passes the right one (`SCALAR`, `U_FIELD`, or `V_FIELD`).

In the code these are named constants, not bare integers, so call sites read as
intent:

```python
from .boundary import SCALAR, U_FIELD, V_FIELD
set_bnd(U_FIELD, u, n)     # negates u across the left/right walls
set_bnd(V_FIELD, v, n)     # negates v across the top/bottom walls
set_bnd(SCALAR, density, n)
```

## Tested explicitly

`tests/test_boundary.py` checks each case directly: scalars equal their
neighbour, normal velocities equal the negation of theirs, tangential components
are copied, and corners are the average. If the convention is ever broken, those
tests fail immediately.

Internal walls (obstacles) extend these same ideas into the middle of the
domain — see [Obstacles & Internal Boundaries](math-obstacles.md).
