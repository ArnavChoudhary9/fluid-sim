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

from ..config import BCType, BoundaryConditions

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


# ---------------------------------------------------------------------------
# Configurable domain boundary conditions (inflow / outflow / periodic / walls)
#
# ``set_bnd`` above is the closed-wall special case. The functions below
# generalise it to per-edge conditions for the CFD scenes (wind tunnels, etc.).
# When every edge is a stationary wall they fall straight back to ``set_bnd`` so
# the default behaviour — and its tests — are byte-for-byte unchanged.
# ---------------------------------------------------------------------------


def _fill_vertical_edge(
    b: int,
    field: np.ndarray,
    n: int,
    *,
    ghost: int,
    inner: int,
    opp: int,
    bctype: BCType,
    inflow_norm: float,
    inflow_tan: float,
    wall_tan: float,
) -> None:
    """Fill one left/right ghost column according to ``bctype`` (u is normal)."""
    rows = slice(1, n + 1)
    if bctype is BCType.WALL:
        if b == U_FIELD:                       # normal component: no through-flow
            field[rows, ghost] = -field[rows, inner]
        elif b == V_FIELD:                     # tangential: free-slip or moving wall
            field[rows, ghost] = (
                field[rows, inner] if wall_tan == 0.0
                else 2.0 * wall_tan - field[rows, inner]
            )
        else:                                  # scalar
            field[rows, ghost] = field[rows, inner]
    elif bctype is BCType.OUTFLOW:
        field[rows, ghost] = field[rows, inner]           # zero-gradient
    elif bctype is BCType.INFLOW:
        if b == U_FIELD:
            field[rows, ghost] = inflow_norm
            field[rows, inner] = inflow_norm              # pin the boundary cell
        elif b == V_FIELD:
            field[rows, ghost] = inflow_tan
            field[rows, inner] = inflow_tan
        else:
            field[rows, ghost] = field[rows, inner]
    else:  # PERIODIC
        field[rows, ghost] = field[rows, opp]


def _fill_horizontal_edge(
    b: int,
    field: np.ndarray,
    n: int,
    *,
    ghost: int,
    inner: int,
    opp: int,
    bctype: BCType,
    inflow_norm: float,
    inflow_tan: float,
    wall_tan: float,
) -> None:
    """Fill one top/bottom ghost row according to ``bctype`` (v is normal)."""
    cols = slice(1, n + 1)
    if bctype is BCType.WALL:
        if b == V_FIELD:                       # normal component
            field[ghost, cols] = -field[inner, cols]
        elif b == U_FIELD:                     # tangential
            field[ghost, cols] = (
                field[inner, cols] if wall_tan == 0.0
                else 2.0 * wall_tan - field[inner, cols]
            )
        else:
            field[ghost, cols] = field[inner, cols]
    elif bctype is BCType.OUTFLOW:
        field[ghost, cols] = field[inner, cols]
    elif bctype is BCType.INFLOW:
        if b == V_FIELD:
            field[ghost, cols] = inflow_norm
            field[inner, cols] = inflow_norm
        elif b == U_FIELD:
            field[ghost, cols] = inflow_tan
            field[inner, cols] = inflow_tan
        else:
            field[ghost, cols] = field[inner, cols]
    else:  # PERIODIC
        field[ghost, cols] = field[opp, cols]


def _set_corners(field: np.ndarray, n: int) -> None:
    field[0, 0] = 0.5 * (field[1, 0] + field[0, 1])
    field[0, n + 1] = 0.5 * (field[1, n + 1] + field[0, n])
    field[n + 1, 0] = 0.5 * (field[n, 0] + field[n + 1, 1])
    field[n + 1, n + 1] = 0.5 * (field[n, n + 1] + field[n + 1, n])


def apply_boundary(b: int, field: np.ndarray, n: int, bc: BoundaryConditions) -> None:
    """Enforce the configured per-edge conditions on a velocity/scalar field.

    Falls back to :func:`set_bnd` when all edges are stationary walls, so the
    closed-wall default is identical (and as fast) as before.
    """
    if bc.is_static_walls:
        set_bnd(b, field, n)
        return

    ux, uy = bc.inflow_velocity
    wl, wr, wt, wb = bc.wall_velocity
    _fill_vertical_edge(b, field, n, ghost=0, inner=1, opp=n,
                        bctype=bc.left, inflow_norm=ux, inflow_tan=uy, wall_tan=wl)
    _fill_vertical_edge(b, field, n, ghost=n + 1, inner=n, opp=1,
                        bctype=bc.right, inflow_norm=ux, inflow_tan=uy, wall_tan=wr)
    _fill_horizontal_edge(b, field, n, ghost=0, inner=1, opp=n,
                         bctype=bc.top, inflow_norm=uy, inflow_tan=ux, wall_tan=wt)
    _fill_horizontal_edge(b, field, n, ghost=n + 1, inner=n, opp=1,
                         bctype=bc.bottom, inflow_norm=uy, inflow_tan=ux, wall_tan=wb)
    _set_corners(field, n)


def apply_pressure_boundary(p: np.ndarray, n: int, bc: BoundaryConditions) -> None:
    """Pressure boundary: Neumann at walls/inflow, ``p = 0`` at outflow, wrap if periodic."""
    if bc.all_pressure_neumann:
        set_bnd(SCALAR, p, n)
        return

    rows = slice(1, n + 1)
    cols = slice(1, n + 1)
    edges = (
        (bc.left, (rows, 0), (rows, 1), (rows, n)),
        (bc.right, (rows, n + 1), (rows, n), (rows, 1)),
        (bc.top, (0, cols), (1, cols), (n, cols)),
        (bc.bottom, (n + 1, cols), (n, cols), (1, cols)),
    )
    for bctype, ghost, inner, opp in edges:
        if bctype is BCType.OUTFLOW:
            p[ghost] = 0.0                      # Dirichlet reference pressure
        elif bctype is BCType.PERIODIC:
            p[ghost] = p[opp]
        else:                                   # WALL / INFLOW → Neumann
            p[ghost] = p[inner]
    _set_corners(p, n)
