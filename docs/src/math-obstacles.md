# Obstacles & Internal Boundaries

Obstacles are solid cells *inside* the domain that the fluid must flow around.
They are the trickiest feature to get right: a naive implementation lets fluid
leak straight through the wall. This chapter explains how each solver primitive
is made obstacle-aware.

## Representation

`FluidState.obstacle` is a boolean mask, `True` where a cell is solid. The brush
sets it via `set_obstacle`, which also evacuates any fluid quantity that was
inside the new wall. Everywhere we talk about *fluid* cells we mean
`~obstacle`.

A solid cell is, in effect, a piece of wall in the middle of the grid — so the
same logic as the domain [boundary conditions](math-boundary-conditions.md)
applies, just at internal faces.

## Three jobs, three primitives

### 1. Velocity at solid faces — reflect the normal component

`apply_obstacle_bnd` looks at each solid cell's four neighbours. For every face
that touches fluid, the velocity component **normal** to that face is reflected
(negated) so the average velocity across the face is zero — no fluid passes
through:

\\[
u_{\text{solid}} = -\,u_{\text{fluid neighbour}}
\quad\text{(for a left/right face)}
\\]

The **tangential** component is either copied (`obstacle_slip = "free"`,
free-slip — fluid glides along the wall) or zeroed (`"no"`, no-slip — fluid sticks
to it). Inside every solid cell, velocity and dye are forced to zero each step so
nothing accumulates within a wall.

### 2. Pressure projection — exclude solids

This is where leaks happen if done carelessly.

> **Pitfall — treating solids as fluid in projection.** The
> [pressure solve](math-projection.md) equalises pressure between neighbouring
> cells. If a solid neighbour is treated like ordinary fluid, pressure equalises
> *through* the wall, and the gradient-subtraction step then drives flow straight
> through it. The obstacle never blocks anything.

The fix imposes a **Neumann condition at solid faces** — zero pressure gradient
across them — by excluding solids from the solve:

- divergence is computed over fluid cells only (solid cells get `div = 0`);
- in each Gauss–Seidel sweep, a cell averages over its **fluid** neighbours only
  and divides by the *count* of fluid neighbours, not by a constant 4:
  \\[
  p_{i,j} \leftarrow
  \frac{\text{div}_{i,j} + \sum_{k\in\text{fluid nbrs}} p_k}{\#\\,\text{fluid nbrs}}
  \\]
- cells with no fluid neighbours (deep inside a wall) are skipped entirely.

`fluid_neighbour_count` precomputes that denominator; `_fluid_neighbour_sum`
(NumPy) and `project_iter` (Numba) compute the masked neighbour sum.

### 3. Advection near solids

The semi-Lagrangian [back-trace](math-advection.md) could sample from inside a
solid. Because solid cells hold copied (scalar) or reflected (velocity) neighbour
values after each boundary pass — rather than raw zeros — the interpolation never
sees an artificial sink, and the field is re-zeroed inside solids afterwards.

## The composite recipe

Putting it together, every solver primitive ends with the same two-stage boundary
pass:

```python
def _field_bnd(self, b, x):
    set_bnd(b, x, n)                               # outer domain walls
    apply_obstacle_bnd(b, x, self.state.obstacle)  # internal solids
```

and projection additionally restricts its divergence, relaxation, and
gradient-subtraction to fluid–fluid faces. Together these guarantee
**no through-flow**: even if the iterative solve is not fully converged, the
reflect pass enforces zero normal velocity at every wall each step.

## Tested

`tests/test_advection.py::test_obstacle_does_not_leak` builds a full vertical
wall, pushes dye and flow into it for many steps from one side, and asserts that
essentially **zero** dye reaches the far side — a direct, physical no-leak check.
A companion test asserts velocity is exactly zero inside solids.

## Free-slip vs no-slip

The `--slip` flag (`SimConfig.obstacle_slip`) chooses the tangential behaviour:

- `free` (default) — fluid slides along walls; gives clean, lively flow-around
  with no artificial boundary layer.
- `no` — fluid sticks to walls; more physically realistic for viscous flow but
  produces a draggy boundary layer.
