# Installation

## Requirements

- **Python 3.11 or newer**
- A desktop environment (the interactive app opens a window via SDL/pygame)

The runtime dependencies are tiny:

| Package | Why |
|---|---|
| `numpy` | all array math and the reference solver |
| `pygame-ce` | window, input, and fast pixel blitting (a maintained fork of pygame) |
| `numba` | *optional* — JIT acceleration for larger grids |

## Install from source

Clone the repository and install it in editable mode. Using a virtual
environment is strongly recommended.

```bash
# from the repository root (d:\fluid-sim)
python -m venv .venv

# activate it
#   Windows (PowerShell):
.venv\Scripts\Activate.ps1
#   macOS / Linux:
source .venv/bin/activate

# install the package and its runtime dependencies
pip install -e .
```

### Optional extras

```bash
# add the Numba-accelerated backend
pip install -e ".[numba]"

# add the development / testing tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

You can combine extras: `pip install -e ".[numba,dev]"`.

## Verify the install without a window

The physics core has **no UI dependency**, so you can confirm everything works
headlessly:

```bash
python examples/headless_demo.py
```

Expected output is a few diagnostics ending in
`core ran with no UI imported — independence confirmed.` If you see that, the
solver is installed and working. (This script also makes a nice template for
batch/offline simulation.)

## Building this documentation

These docs are written as an [mdBook](https://rust-lang.github.io/mdBook/). The
Markdown in `docs/src/` is perfectly readable as-is, but to render the book:

```bash
# install mdBook (a single Rust binary) — see the mdBook site for options
cargo install mdbook        # or download a release binary

# build and open the book
mdbook serve docs --open
```

Math is rendered with MathJax (enabled in `docs/book.toml`).
