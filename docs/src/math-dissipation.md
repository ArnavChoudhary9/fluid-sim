# Numerical Dissipation & Its Limits

Stable Fluids buys unconditional stability with one currency: **numerical
dissipation**. Understanding it explains both why the simulation looks the way it
does and what you can do about it.

## Where it comes from

Every step, [semi-Lagrangian advection](math-advection.md) reads each new value by
**bilinearly interpolating** from the old field. Interpolation is a weighted
average, and averaging neighbouring values is exactly a low-pass (smoothing)
filter. So even with viscosity and dye-diffusion set to zero, the act of
advecting *itself* blurs sharp features a little every step.

Two visible consequences:

- **Dye fades and softens.** Crisp ink edges become fuzzy; fine filaments wash
  out over time. This is *numerical* diffusion, distinct from the physical
  \\(\kappa\nabla^2 d\\) term.
- **Swirl decays.** Vorticity (the curl of velocity) is smeared by the same
  mechanism, so eddies gently spin down even with no physical viscosity. The
  fluid is effectively more viscous than `viscosity` alone would suggest.

It is grid-dependent: a finer grid (`--grid 256`) interpolates over smaller cells,
so it dissipates less and holds detail longer — at a higher compute cost.

## Why we accept it

The trade is deliberate and worth it for interactive use:

- **Unconditional stability.** The same interpolation that smears is what
  guarantees a back-traced value can never exceed the existing range, so the
  simulation cannot blow up no matter how hard you fling the mouse.
- **Simplicity.** The base method is short, easy to read, and easy to reason
  about — in keeping with the project's goals.

A small explicit `fade` factor (`SimConfig.fade`, default `0.999`) is applied to
the dye on top of this, so a long session does not saturate the whole screen to
white. It is multiplicative and barely perceptible per step.

## Softening it (deliberately left as extensions)

Several standard techniques reduce dissipation. They are **not** in the base
solver — keeping it minimal is intentional — but each slots cleanly into the
existing structure and is a good exercise (see
[Extending the Simulation](extending.md)):

| Technique | Idea | Cost |
|---|---|---|
| **MacCormack / BFECC advection** | advect forward then backward to estimate and cancel the interpolation error | ~2–3× advection cost |
| **Vorticity confinement** | detect where vorticity is being lost and add a force that re-injects it | one extra force term |
| **Higher-order interpolation** (e.g. cubic) | a sharper interpolation kernel smears less | more samples per cell |
| **Finer grid** | smaller cells, less averaging | quadratic in `n` |

> **Pitfall — chasing zero dissipation.** Pushing too hard the other way
> (very high-order or anti-dissipative schemes) reintroduces the instabilities
> Stam's method was designed to avoid: overshoots, ringing, and negative dye.
> The art is a controlled amount of dissipation — enough to stay stable, little
> enough to look alive.

## Summary

The fluid you see is shaped as much by numerical dissipation as by the physical
viscosity and diffusion coefficients. If detail fades faster than you'd like,
raise `--grid` first; if you want to go further, MacCormack advection or vorticity
confinement are the natural next steps.
