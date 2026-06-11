"""Shape primitives rasterise to correct interior masks."""

from __future__ import annotations

import numpy as np

from fluidsim.scenes.shapes import Circle, Ellipse, Plate, Rectangle, rasterize


def test_circle_centre_and_area() -> None:
    n = 120
    mask = Circle(0.5, 0.5, 0.25).mask(n)
    assert mask.shape == (n, n)
    assert mask[n // 2, n // 2]          # centre is solid
    assert not mask[0, 0]                # corner is empty
    # Filled fraction ≈ π r².
    assert abs(mask.mean() - np.pi * 0.25 ** 2) < 0.01


def test_rectangle_bounds() -> None:
    n = 100
    mask = Rectangle(0.25, 0.25, 0.75, 0.75).mask(n)
    assert mask[n // 2, n // 2]
    assert not mask[0, 0]
    assert abs(mask.mean() - 0.25) < 0.02


def test_ellipse_is_anisotropic() -> None:
    n = 100
    mask = Ellipse(0.5, 0.5, 0.4, 0.1).mask(n)
    # Wide in x, thin in y: a horizontal neighbour is inside, a vertical one is not.
    assert mask[n // 2, int(0.8 * n)]
    assert not mask[int(0.8 * n), n // 2]


def test_plate_rotation_changes_orientation() -> None:
    n = 100
    flat = Plate(0.5, 0.5, 0.6, 0.05, angle=0.0).mask(n)
    vert = Plate(0.5, 0.5, 0.6, 0.05, angle=90.0).mask(n)
    assert flat.sum() > 0 and vert.sum() > 0
    assert not np.array_equal(flat, vert)


def test_rasterize_unions_shapes() -> None:
    n = 80
    a = Circle(0.3, 0.5, 0.1)
    b = Circle(0.7, 0.5, 0.1)
    union = rasterize((a, b), n)
    assert union.sum() == a.mask(n).sum() + b.mask(n).sum()  # disjoint discs
