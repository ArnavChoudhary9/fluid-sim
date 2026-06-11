# Controls & Usage

The window title bar always shows the live status: whether the sim is
`running`/`paused`, the current brush size, and whether the overlay is on.

## Mouse

| Action | Effect |
|---|---|
| **Left-drag** | Inject dye **and** push the fluid in the direction you move. Hold still to dab dye without pushing; drag fast to fling it. |
| **Right-drag** | Paint **solid obstacles** the fluid flows around. |
| **Middle-drag** | **Erase** obstacles. |

The colour you inject is the brush's current palette colour (cycle it with the
keyboard). The size of every brush stroke — dye, force, and obstacles — is the
current brush radius.

## Keyboard

| Key | Action |
|---|---|
| `Space` | Pause / resume the physics (you can still place obstacles while paused) |
| `C` | Clear all dye and velocity (obstacles are kept) |
| `V` | Toggle the velocity-field overlay (arrows showing flow direction/speed) |
| `Tab` or `]` | Next dye colour |
| `[` | Previous dye colour |
| `+` / `=` | Grow the brush |
| `-` | Shrink the brush |
| `Esc` | Quit |

> The physical-key → action binding lives in exactly one place
> (`app/pygame_backend.py`). Rebinding a key is a one-line change there and
> affects nothing else.

## Things to try

- **Smoke plume.** Pick a warm colour, dab dye near the bottom, then give gentle
  upward flicks — watch it billow and curl.
- **Flow around a pillar.** Right-drag a vertical bar in the middle, then
  left-drag a steady stream at it from one side. The dye splits and forms a
  wake. Toggle the overlay (`V`) to see the recirculation.
- **Colour mixing.** Inject one colour, cycle with `Tab`, inject another across
  it, and watch them advect and blend.
- **Viscosity feel.** Relaunch with `--viscosity 1e-4` for a syrupy fluid versus
  the near-inviscid default.

## Why fixed-timestep?

The simulation advances in fixed `dt` increments (default 1/60 s) regardless of
your actual frame rate, using a time accumulator. This keeps the behaviour
**deterministic and machine-independent** — the same drag produces the same swirl
whether your machine renders at 30 or 120 FPS. The mechanism is described in
[Stable Fluids: The Algorithm](math-stable-fluids.md) and
[Module Reference](architecture-modules.md).
