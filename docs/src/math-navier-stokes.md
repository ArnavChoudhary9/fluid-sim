# Navier–Stokes for Incompressible Flow

Everything the solver does is an attempt to integrate the **incompressible
Navier–Stokes equations** forward in time. This chapter states them, explains
each term physically, and sets up the splitting that the next chapters implement.

## The equations

For a fluid with velocity field \\( \mathbf{u}(\mathbf{x}, t) \\) and constant
density, the incompressible Navier–Stokes equations are:

\\[
\frac{\partial \mathbf{u}}{\partial t}
= -(\mathbf{u}\cdot\nabla)\mathbf{u}
\\; - \\; \frac{1}{\rho}\nabla p
\\; + \\; \nu\,\nabla^2 \mathbf{u}
\\; + \\; \mathbf{f}
\\]

\\[
\nabla \cdot \mathbf{u} = 0
\\]

The first is conservation of momentum (Newton's second law for a fluid parcel);
the second is the **incompressibility constraint**.

## What each term means

Reading the momentum equation term by term — this is the entire physics of the
simulation:

| Term | Name | Physical meaning | Implemented by |
|---|---|---|---|
| \\(-(\mathbf{u}\cdot\nabla)\mathbf{u}\\) | **advection** | the fluid carries its own velocity along with it | [advect](math-advection.md) |
| \\(-\frac{1}{\rho}\nabla p\\) | **pressure** | pressure pushes fluid from high to low to prevent compression | [project](math-projection.md) |
| \\(\nu\,\nabla^2\mathbf{u}\\) | **diffusion / viscosity** | internal friction smears velocity differences out | [diffuse](math-diffusion.md) |
| \\(\mathbf{f}\\) | **external force** | gravity, or *your mouse* | `add_velocity` |

\\(\nu\\) is the **kinematic viscosity** (`SimConfig.viscosity`). Small \\(\nu\\)
gives a thin, swirly fluid; large \\(\nu\\) gives a thick, syrupy one.

## The incompressibility constraint

\\( \nabla \cdot \mathbf{u} = 0 \\) says the velocity field has **zero
divergence**: as much fluid flows into any region as flows out, so nothing is
compressed or created. This is what makes fluid swirl instead of simply blurring
away, and enforcing it (the *projection* step) is the mathematical heart of the
method. See [Projection](math-projection.md).

## Carrying dye

We also want to *see* the fluid. We advect a dye (smoke) concentration
\\( d(\mathbf{x}, t) \\) — actually three of them, one per colour channel —
through the same velocity field:

\\[
\frac{\partial d}{\partial t}
= -(\mathbf{u}\cdot\nabla)d
\\; + \\; \kappa\,\nabla^2 d
\\; + \\; S
\\]

with diffusion coefficient \\(\kappa\\) (`SimConfig.diffusion`) and a source
\\(S\\) (your mouse, via `add_dye`). The dye is a *passive scalar*: it is pushed
around by the fluid but does not affect it.

## The plan of attack: operator splitting

We never solve the full coupled system at once. Instead, each timestep we apply
the terms **one at a time**, in sequence — *operator splitting*. Each sub-step is
simple, stable, and individually solvable; together they advance the fluid. The
ordering matters and is the subject of the [next chapter](math-stable-fluids.md).
