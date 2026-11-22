
"""Tests del solver tridiagonal."""

import numpy as np
import pytest

from minipde.linear_solvers import solve_tridiagonal


def test_thomas_vs_numpy():
    """Comparar Thomas con NumPy."""
    A = np.array([
        [4.0, -1.0, 0.0],
        [-1.0, 4.0, -1.0],
        [0.0, -1.0, 4.0],
    ])
    b = np.array([1.0, 2.0, 3.0])

    u_thomas = solve_tridiagonal(A, b)
    u_numpy = np.linalg.solve(A, b)

    np.testing.assert_allclose(u_thomas, u_numpy, atol=1e-12)


def test_non_tridiagonal():
    """Comprobar que se rechazan matrices no tridiagonales."""
    A = np.array([
        [4.0, 1.0, 1.0],
        [1.0, 4.0, 1.0],
        [1.0, 1.0, 4.0],
    ])
    b = np.ones(3)

    with pytest.raises(ValueError):
        solve_tridiagonal(A, b)
