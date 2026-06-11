# Advection (Semi-Lagrangian)

**Advection** moves a quantity along the velocity field: the fluid carries its
own velocity, and it carries the dye. This is the term
\\( -(\mathbf{u}\cdot\nabla)\,q \\) for any quantity \\(q\\).

## The unstable way (and why we avoid it)

The obvious discretisation is *forward* (Eulerian): estimate the spatial
gradient and step \\(q\\) forward by \\(-\mathbf{u}\cdot\nabla q\\, \Delta t\\).
This is only conditionally stable — if a parcel would move more than one cell per
step (the CFL condition \\( |\mathbf{u}|\,\Delta t / h > 1 \\)), it overshoots and
the simulation explodes. For interactive use, where the user can fling the fluid
arbitrarily fast, that is fatal.

## Stam's trick: trace backwards

Stam's **semi-Lagrangian** method is unconditionally stable. Instead of pushing
values forward, it asks, for each grid cell:

> *Where was the fluid that is now here, one timestep ago?*

It traces that point **backwards** along the velocity field and reads the old
value there by interpolation:

\\[
\mathbf{x}_{\text{back}} = \mathbf{x} - \Delta t\,\mathbf{u}(\mathbf{x}),
\qquad
q^{\,\text{new}}(\mathbf{x}) = q^{\,\text{old}}(\mathbf{x}_{\text{back}})
\\]

Because the new value is *interpolated from existing values*, it can never exceed
the current range — no overshoot, no blow-up, for any timestep.

## Bilinear interpolation

The back-traced point \\(\mathbf{x}_{\text{back}}\\) lands between grid cells, so
we **bilinearly interpolate** from the four surrounding cells. With the cell
corner at integer \\((i_0, j_0)\\) and fractional offsets \\(s, t \in [0,1)\\):

\\[
q(\mathbf{x}_{\text{back}}) =
(1-t)\big[(1-s)\,q_{i_0,j_0} + s\,q_{i_0,j_0+1}\big]
+ t\big[(1-s)\,q_{i_0+1,j_0} + s\,q_{i_0+1,j_0+1}\big]
\\]

## In the code

`NumpySolver._advect` is a vectorised version of exactly this. Using `dt0 = dt·n`
(the velocity scaling that matches our \\(h=1\\) grid):

```python
x = self._xs - dt0 * vel_u[1:-1, 1:-1]   # back-trace x for every cell at once
y = self._ys - dt0 * vel_v[1:-1, 1:-1]
np.clip(x, 0.5, n + 0.5, out=x)          # keep samples inside the real grid
np.clip(y, 0.5, n + 0.5, out=y)

x0 = np.floor(x).astype(int); x1 = x0 + 1
y0 = np.floor(y).astype(int); y1 = y0 + 1
sx1 = x - x0; sx0 = 1 - sx1              # bilinear weights
sy1 = y - y0; sy0 = 1 - sy1

d[1:-1, 1:-1] = ( sy0*(sx0*d0[y0,x0] + sx1*d0[y0,x1])
                + sy1*(sx0*d0[y1,x0] + sx1*d0[y1,x1]) )
```

The Numba backend (`_numba_kernels.advect_scalar` / `advect_rgb`) does the same
arithmetic in an explicit loop. The RGB version simply repeats the interpolation
over the three colour channels.

> **Pitfall — clamping the back-trace.** The line `np.clip(x, 0.5, n + 0.5)` is
> essential. Without it, a fast velocity could trace to a point outside the array
> and crash, or read garbage from the ghost border. Clamping keeps every sample
> inside the valid region. Near obstacles, solid cells hold copied/reflected
> neighbour values (see [Obstacles](math-obstacles.md)) so the interpolation
> never pulls a value from "inside a wall".

## The cost: it smears

Semi-Lagrangian advection is stable but **dissipative**: each interpolation
slightly averages neighbouring values, so sharp dye edges blur a little every
step. This is the price of stability, examined in
[Numerical Dissipation](math-dissipation.md).
