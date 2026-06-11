# Rendering to Video

`fluidsim.recording` steps a [scene](scenes.md), [visualises](visualization.md)
each frame, and encodes the result — an MP4 (via the optional `imageio` +
`imageio-ffmpeg` dependency) or, as a zero-config fallback, a numbered PNG
sequence. It is the offline counterpart to the interactive window: fully
headless, no pygame.

## Install the video extra

```bash
pip install -e ".[video]"
```

This pulls in `imageio` and `imageio-ffmpeg` (a bundled ffmpeg — no system
install needed). `imageio` is imported **lazily** inside the encoder, so
importing `fluidsim` never requires it.

## The CLI

```bash
fluidsim-render <scene> <output> [options]
```

```bash
# Vorticity view of the wind tunnel → MP4
fluidsim-render wind_tunnel tunnel.mp4 --view vorticity --frames 600

# Smoke plume at a custom size and grid
fluidsim-render smoke_plume plume.mp4 --size 1280 720 --grid 220

# Shear layer with streamlines → a PNG sequence (output has no video suffix)
fluidsim-render shear_layer frames/ --overlay streamlines
```

Key options (`fluidsim-render --help` for all): `--view`, `--colormap`,
`--overlay`, `--size W H`, `--frames`, `--fps`, `--warmup` (settling steps before
recording), `--steps-per-frame` (speed up playback), `--grid`, `--backend`.

## The Python API

```python
from fluidsim.recording import render_scene
from fluidsim.render.visualize import VisualConfig
from fluidsim.scenes.library import wind_tunnel

render_scene(
    wind_tunnel(n=200),
    "tunnel.mp4",
    frames=400,
    fps=60,
    warmup=60,                                  # let the wake develop first
    visual=VisualConfig(view="vorticity", output_size=(960, 960)),
)
```

`render_named_scene("wind_tunnel", "out.mp4", ...)` is a convenience that builds a
built-in scene by name. See `examples/render_video.py` and
`examples/custom_scene.py`.

## Output formats and the fallback

- The output **suffix** decides the encoder: `.mp4`, `.mov`, `.webm`, `.mkv`, or
  `.gif` → video; anything else (e.g. a directory name) → a **PNG sequence**
  written as `frame_00000.png`, `frame_00001.png`, ….
- If video encoding fails (for example ffmpeg is unavailable), the recorder
  **falls back to a PNG sequence** in a sibling directory and tells you so, rather
  than aborting a long run.

> **Tip — even dimensions for MP4.** H.264 prefers even width/height. The square
> defaults (e.g. `1080×1080`) are fine; if you pick custom sizes, keep them even.

## Performance

- The grid resolution (`--grid`) drives simulation cost (\\(\sim n^2\\)); the
  output resolution (`--size`) only affects the cheap upscale/encode. Render a
  modest grid to a large frame.
- Use `--backend numba` for larger grids (see [Performance](reference-performance.md)).
- `--warmup` runs steps without encoding; `--steps-per-frame > 1` advances more
  simulation time per recorded frame (faster-looking playback, fewer frames to
  encode).
