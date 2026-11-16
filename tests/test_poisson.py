
"""Tests for the one-dimensional Poisson solver."""

import numpy as np

from minipde.mesh import create_mesh_1d
from minipde.poisson import assemble_poisson_1d, solve_poisson_1d


def test_poisson_manufactured():
    """Compare numerical and exact solutions."""
    x = create_mesh_1d(0, 1, 81)

    exact = np.sin(np.pi * x)
    source = lambda x: np.pi**2 * np.sin(np.pi * x)

    numerical = solve_poisson_1d(x, source)
    error = np.max(np.abs(numerical - exact))

    assert error < 1e-3


def test_poisson_dirichlet():
    """Check nonzero Dirichlet boundary conditions."""
    x = create_mesh_1d(0, 1, 21)

    numerical = solve_poisson_1d(
        x, np.zeros_like(x), left=2.0, right=5.0
    )

    exact = 2.0 + 3.0 * x
    np.testing.assert_allclose(numerical, exact, atol=1e-12)


def test_poisson_residual():
    """Check the algebraic residual."""
    x = create_mesh_1d(0, 1, 41)

    source = lambda x: np.pi**2 * np.sin(np.pi * x)

    A, b = assemble_poisson_1d(x, source)
    numerical = np.linalg.solve(A, b)

    residual = np.linalg.norm(A @ numerical - b, ord=np.inf)

    assert residual < 1e-9
