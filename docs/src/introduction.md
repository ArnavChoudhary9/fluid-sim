# Introduction

**fluidsim** is an interactive, real-time **2D fluid simulation** written in
Python. You paint coloured dye with the mouse, push the fluid around by dragging,
and drop solid obstacles for the flow to swirl around — all while a physically
based solver integrates the incompressible Navier–Stokes equations underneath.

![concept](https://img.shields.io/badge/method-Stable%20Fluids-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-AGPL--3.0-green)

## What it does

- Simulates an **incompressible fluid** on a regular grid using Jos Stam's
  *Stable Fluids* method — unconditionally stable, so it never "blows up".
- Carries **coloured dye** (three advected channels) through the velocity field,
  so smoke and ink mix and transport realistically.
- Lets you **interact live**: left-drag to inject dye and force, right-drag to
  build walls, middle-drag to erase them, plus keyboard niceties.
- Ships a **pure-NumPy** solver and an **optional Numba-accelerated** one behind
  one interface, selectable at runtime with graceful fallback.

## Why this design

The project was built to three explicit principles, and the documentation keeps
returning to them:

1. **Modular and independent.** Each part does exactly one thing and could be
   reused or replaced on its own.
2. **Principle of least privilege.** The physics core cannot see the UI; the
   renderer can only *read* the simulation; input can only *inject* through a
   narrow API. These boundaries are enforced by import direction and even by a
   [unit test](architecture-layering.md).
3. **Clean, understandable code.** Small files, named constants instead of magic
   numbers, and docstrings that explain *why*, not just *what*.

## How to read this book

- New here? Start with [Installation](getting-started-install.md) and
  [Running the Simulation](getting-started-running.md), then learn the
  [Controls](usage-controls.md).
- Want to understand the structure? See
  [Design Overview & Decisions](architecture-overview.md).
- Curious about the physics? The [Mathematics](math-navier-stokes.md) section
  derives every step from the Navier–Stokes equations down to the discrete
  stencils actually in the code.
- Extending or contributing? See [Extending the Simulation](extending.md) and the
  [Configuration Reference](reference-configuration.md).

> Throughout, **pitfall callouts** like this one flag the subtle mistakes that
> make a fluid solver leak, stick, or explode — and how this code avoids them.
