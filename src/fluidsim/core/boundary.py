"""Boundary conditions: domain walls and internal obstacles.

This is the highest-risk code in the solver. Two distinct jobs live here:

1. :func:`set_bnd` — the outer *domain walls* (the ghost ring). The sign
   convention differs between scalars and the two velocity components, and
   getting it wrong produces fluid that leaks through the window edges or
   sticks to them.

2. :func:`apply_obstacle_bnd` and :func:`zero_inside_obstacles` — *internal
   solids*. A solid cell must reflect the velocity component normal to each
   fluid-facing face (so nothing flows through it) while leaving the tangential
   component free (free-slip) or zeroed (no-slip).

The ``b`` flag selects which field we are fixing up:

* ``b = 0`` — a **scalar** (dye, pressure, divergence): ghost = neighbour
  (homogeneous Neumann, ``d/dn = 0``).
* ``b = 1`` — the **x-velocity** ``u``: negated across the left/right walls.
* ``b = 2`` — the **y-velocity** ``v``: negated across the top/bottom walls.

Everything here operates on the padded ``(n + 2, n + 2)`` arrays described in
:mod:`fluidsim.core.grid`.
"""

from __future__ import annotations

import numpy as np

# b-flag constants, named so call sites read as intent rather than magic numbers.
SCALAR = 0
U_FIELD = 1
V_FIELD = 2


def set_bnd(b: int, field: np.ndarray, n: int) -> None:
    """Enforce wall conditions on the ghost ring of a ``(n + 2, n + 2)`` field.

    Parameters
    ----------
    b:
        One of :data:`SCALAR`, :data:`U_FIELD`, :data:`V_FIELD`.
    field:
        The padded array to fix up, modified in place.
    n:
        Interior grid size.
    """
    # Left / right walls (vary in y, fixed x). For u (b == 1) the *normal*
    # component is x, so we negate; otherwise we copy the neighbour.
    sign_x = -1.0 if b == U_FIELD else 1.0
    field[1 : n + 1, 0] = sign_x * field[1 : n + 1, 1]
    field[1 : n + 1, n + 1] = sign_x * field[1 : n + 1, n]

    # Top / bottom walls (vary in x, fixed y). For v (b == 2) the normal
    # component is y, so we negate; otherwise we copy.
    sign_y = -1.0 if b == V_FIELD else 1.0
    field[0, 1 : n + 1] = sign_y * field[1, 1 : n + 1]
    field[n + 1, 1 : n + 1] = sign_y * field[n, 1 : n + 1]

    # Corners take the average of their two edge neighbours.
    field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
    field[0, n + 1] = 0.5 * (field[1, n + 1] + field[0, n])
    field[n + 1, 0] = 0.5 * (field[n, 0] + field[n + 1, 1])
    field[n + 1, n + 1] = 0.5 * (field[n, n + 1] + field[n + 1, n])


def zero_inside_obstacles(field: np.ndarray, obstacle: np.ndarray) -> None:
    """Force a field to zero inside every solid cell (in place).

    Used to keep momentum and dye from accumulating inside walls.
    """
    field[obstacle] = 0.0


def apply_obstacle_bnd(
    b: int,
    field: np.ndarray,
    obstacle: np.ndarray,
    *,
    slip: str = "free",
) -> None:
    """Enforce solid-wall conditions at internal obstacle faces (in place).

    For each solid cell we look at its four neighbours. A velocity component
    that is *normal* to a fluid-facing face is reflected (negated) so the
    average velocity across that face is zero — no fluid passes through the
    wall. Scalars simply copy the adjacent fluid value (Neumann), and the
    tangential velocity component is either copied (free-slip) or zeroed
    (no-slip).

    Parameters
    ----------
    b:
        Field selector (:data:`SCALAR` / :data:`U_FIELD` / :data:`V_FIELD`).
    field:
        Padded array, modified in place.
    obstacle:
        Boolean solid mask, same shape as ``field``.
    slip:
        ``"free"`` or ``"no"`` — tangential treatment for velocity fields.
    """
    solid = obstacle
    if not solid.any():
        return

    # Neighbour fluid masks. ``fluid_right`` is True for a solid cell whose
    # right neighbour is fluid, etc. Shifting the (negated) solid mask tells us
    # where each fluid-facing face lives. Borders are treated as solid so we
    # never index past the array edge.
    fluid = ~solid
    fl = np.zeros_like(solid)
    fr = np.zeros_like(solid)
    fu = np.zeros_like(solid)
    fd = np.zeros_like(solid)
    fl[:, 1:] = solid[:, 1:] & fluid[:, :-1]   # fluid to the left
    fr[:, :-1] = solid[:, :-1] & fluid[:, 1:]  # fluid to the right
    fu[1:, :] = solid[1:, :] & fluid[:-1, :]   # fluid above
    fd[:-1, :] = solid[:-1, :] & fluid[1:, :]  # fluid below

    if b == U_FIELD:
        # x is normal across left/right faces -> reflect from that neighbour.
        field[fr] = -np.roll(field, -1, axis=1)[fr]
        field[fl] = -np.roll(field, 1, axis=1)[fl]
        # x is tangential across top/bottom faces.
        if slip == "free":
            field[fd] = np.roll(field, -1, axis=0)[fd]
            field[fu] = np.roll(field, 1, axis=0)[fu]
        else:
            field[fd | fu] = 0.0
    elif b == V_FIELD:
        # y is normal across top/bottom faces -> reflect from that neighbour.
        field[fd] = -np.roll(field, -1, axis=0)[fd]
        field[fu] = -np.roll(field, 1, axis=0)[fu]
        # y is tangential across left/right faces.
        if slip == "free":
            field[fr] = np.roll(field, -1, axis=1)[fr]
            field[fl] = np.roll(field, 1, axis=1)[fl]
        else:
            field[fr | fl] = 0.0
    else:  # SCALAR: copy the adjacent fluid value (Neumann), prefer one face.
        field[fr] = np.roll(field, -1, axis=1)[fr]
        field[fl] = np.roll(field, 1, axis=1)[fl]
        field[fd] = np.roll(field, -1, axis=0)[fd]
        field[fu] = np.roll(field, 1, axis=0)[fu]


def fluid_neighbour_count(obstacle: np.ndarray) -> np.ndarray:
    """Count fluid neighbours of every cell (used by the obstacle-aware solve).

    Returns an ``int8`` array where each interior cell holds the number of its
    four orthogonal neighbours that are fluid. Cells with zero fluid neighbours
    are skipped by the pressure relaxation (they are deep inside a solid).
    """
    fluid = (~obstacle).astype(np.int8)
    count = np.zeros_like(fluid)
    count[1:-1, 1:-1] = (
        fluid[1:-1, :-2]
        + fluid[1:-1, 2:]
        + fluid[:-2, 1:-1]
        + fluid[2:, 1:-1]
    )
    return count
