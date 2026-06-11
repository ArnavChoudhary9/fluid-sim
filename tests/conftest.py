"""Shared pytest fixtures.

The ``backend`` fixture parametrises tests over every solver backend that is
actually importable: always ``"numpy"``, and ``"numba"`` only when numba is
installed. Numba-only tests can depend on it and will simply not run the numba
case when it is absent.
"""

from __future__ import annotations

import pytest

from fluidsim.core.numba_solver import NUMBA_AVAILABLE

_BACKENDS = ["numpy"] + (["numba"] if NUMBA_AVAILABLE else [])


@pytest.fixture(params=_BACKENDS)
def backend(request) -> str:
    return request.param
