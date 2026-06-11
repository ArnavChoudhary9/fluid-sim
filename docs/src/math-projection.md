# Projection & the Pressure Poisson Equation

**Projection** is the mathematical heart of an incompressible solver. It takes a
velocity field that has drifted away from
\\( \nabla\cdot\mathbf{u} = 0 \\) and removes its divergence, so the fluid swirls
instead of compressing. Get this wrong and the fluid leaks, collapses, or
explodes.

## The Helmholtz–Hodge decomposition

Any velocity field \\(\mathbf{w}\\) splits uniquely into a divergence-free part
\\(\mathbf{u}\\) and the gradient of a scalar field \\(p\\):

\\[
\mathbf{w} = \mathbf{u} + \nabla p,
\qquad \nabla\cdot\mathbf{u} = 0
\\]

To recover the divergence-free \\(\mathbf{u}\\) we just subtract \\(\nabla p\\) —
but first we need \\(p\\). Taking the divergence of the decomposition and using
\\(\nabla\cdot\mathbf{u}=0\\):

\\[
\nabla\cdot\mathbf{w} = \nabla\cdot\nabla p = \nabla^2 p
\\]

So \\(p\\) (which plays the role of **pressure**) solves a **Poisson equation**
with the field's divergence as the right-hand side:

\\[
\boxed{\\; \nabla^2 p = \nabla\cdot\mathbf{w} \\;}
\\]

## The three steps of `project`

1. **Compute divergence** of the current velocity:
   \\[
   \text{div}_{i,j} = -\tfrac{1}{2}\big(u_{i,j+1}-u_{i,j-1} + v_{i+1,j}-v_{i-1,j}\big)
   \\]
2. **Solve the Poisson equation** \\(\nabla^2 p = \text{div}\\) for the pressure,
   using the same red-black Gauss–Seidel sweeps as
   [diffusion](math-diffusion.md):
   \\[
   p_{i,j} \leftarrow \tfrac{1}{4}\big(\text{div}_{i,j} + p_{i-1,j}+p_{i+1,j}+p_{i,j-1}+p_{i,j+1}\big)
   \\]
3. **Subtract the pressure gradient** to make the field divergence-free:
   \\[
   u_{i,j} \mathrel{-}= \tfrac{1}{2}(p_{i,j+1}-p_{i,j-1}),
   \qquad
   v_{i,j} \mathrel{-}= \tfrac{1}{2}(p_{i+1,j}-p_{i-1,j})
   \\]

The grid spacing \\(h\\) cancels between the divergence and the gradient, so the
code drops it. This maps directly onto `NumpySolver._project`.

## An honest subtlety: it is a contraction, not a one-shot

> **Pitfall — the collocated central-difference scheme is not exactly
> divergence-free, and projection is not idempotent.** The divergence in step 1
> and the gradient in step 3 both use *central* differences (spacing 2), while the
> Poisson solve in step 2 uses the *narrow* 5-point Laplacian (spacing 1). These
> two stencils are not algebraically inverse, so a single projection leaves a
> small residual divergence (concentrated in high-frequency "checkerboard"
> modes), and projecting again changes the field a little more.
>
> What *is* true — and what the code relies on — is that projection is a
> **contraction**: each pass reduces the divergence and never increases it. This
> is precisely why the velocity step projects **twice** (see
> [Stable Fluids](math-stable-fluids.md)) rather than trusting one pass to be
> exact. The test `test_reprojection_is_a_contraction` asserts this property, and
> `test_projection_removes_divergence` asserts a single pass already removes the
> vast majority of the divergence.

This is a well-known property of Stam's *collocated* (all quantities at cell
centres) scheme. Staggered ("MAC") grids avoid the decoupling but complicate
everything else; for a real-time visual simulation the collocated scheme is the
right trade-off.

## Convergence and iteration count

Gauss–Seidel converges *slowly* for the Poisson equation — the error decays by a
factor like \\(1 - O(1/n^2)\\) per sweep, so fully converging an \\(n\times n\\)
grid would take \\(O(n^2)\\) sweeps. The default 20 sweeps is nowhere near
converged, and that is fine: combined with re-projection every step, it keeps the
fluid visually incompressible. Increasing `--iterations` tightens it at a cost in
frame time. (The projection unit tests use a small grid and many sweeps precisely
so that convergence is observable.)

Obstacles change the Poisson solve so that fluid flows *around* solids instead of
through them — the subject of [Obstacles & Internal Boundaries](math-obstacles.md).
