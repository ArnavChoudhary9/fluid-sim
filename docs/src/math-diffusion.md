# Diffusion (Implicit Gauss–Seidel)

**Diffusion** is the term \\( \nu\,\nabla^2 \mathbf{u} \\) (for velocity) and
\\( \kappa\,\nabla^2 d \\) (for dye). Physically it is **viscosity** smearing
velocity differences out, and dye slowly blurring. Mathematically it is the heat
equation.

## Why not explicit?

The naive explicit update,
\\( q^{\,\text{new}} = q + \Delta t\,\nu\,\nabla^2 q \\),
is only stable when \\( \Delta t\,\nu / h^2 \\) is small. For a viscous fluid or a
large timestep it explodes — the same conditional-stability trap as forward
advection.

## Implicit (backward Euler) instead

Stam evaluates the Laplacian at the **new** time, giving a backward-Euler step
that is stable for *any* \\(\Delta t\\) and any \\(\nu\\):

\\[
q^{\,\text{new}} - \Delta t\,\nu\,\nabla^2 q^{\,\text{new}} = q^{\,\text{old}}
\\]

Discretising the Laplacian with the 5-point stencil (a cell minus the average of
its four neighbours) and writing \\( a = \Delta t\,\nu\,n^2 \\):

\\[
(1 + 4a)\,q^{\,\text{new}}_{i,j}
- a\big(q^{\,\text{new}}_{i-1,j} + q^{\,\text{new}}_{i+1,j}
      + q^{\,\text{new}}_{i,j-1} + q^{\,\text{new}}_{i,j+1}\big)
= q^{\,\text{old}}_{i,j}
\\]

This is a large sparse linear system \\( A\,q^{\text{new}} = q^{\text{old}} \\).
We solve it iteratively.

## Gauss–Seidel relaxation

Rearranging for the centre cell gives an update we can sweep repeatedly until it
settles:

\\[
q_{i,j} \leftarrow
\frac{q^{\,\text{old}}_{i,j} + a\,(q_{i-1,j}+q_{i+1,j}+q_{i,j-1}+q_{i,j+1})}{1 + 4a}
\\]

A fixed number of sweeps (`SimConfig.iterations`, default 20) is "good enough" for
visual purposes — full convergence is unnecessary.

## Red-black ordering (so it vectorises)

> **Pitfall — true Gauss–Seidel does not vectorise.** Classic Gauss–Seidel
> updates cells in sequence, each using neighbours *already updated this sweep* —
> an inherently serial dependency that NumPy cannot express as one array
> operation.
>
> The fix is **red-black ordering**. Colour the grid like a checkerboard. A red
> cell's four neighbours are all black and vice-versa, so we can update *all* red
> cells at once (they don't depend on each other), then all black cells at once.
> This recovers Gauss–Seidel-style convergence while staying fully vectorised.

In `NumpySolver._lin_solve`:

```python
for _ in range(self.config.iterations):
    x[self._red]   = ((x0 + a * self._neighbour_sum(x)) / c)[self._red]
    x[self._black] = ((x0 + a * self._neighbour_sum(x)) / c)[self._black]
    self._field_bnd(b, x)        # re-apply boundaries after each sweep
```

with \\( c = 1 + 4a \\). The same `_lin_solve` machinery is reused for the
pressure solve in [Projection](math-projection.md) (there with \\(a=1\\) and a
variable, obstacle-aware denominator). The Numba backend uses the identical
red-black ordering in an explicit loop, which is why the two backends agree.

## A note on the defaults

The default viscosity and diffusion are tiny (`1e-6`), so the fluid is nearly
inviscid and the dye barely blurs from diffusion alone — most of the smearing you
see actually comes from advection ([Dissipation](math-dissipation.md)). Crank
`--viscosity 1e-4` to feel a genuinely thick fluid.
