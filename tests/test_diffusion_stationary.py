
"""Tests de difusion estacionaria 1D."""

import numpy as np

from minipde.mesh import create_mesh_1d
from minipde.models.diffusion_stationary import (
    assemble_diffusion_stationary_1d,
    solve_diffusion_stationary_1d,
)


def test_assembly():
    """Comprobar el ensamblaje de la matriz."""
    x = create_mesh_1d(0, 1, 3)

    A, b = assemble_diffusion_stationary_1d(
        x, 2.0, 0.0, 100.0, 2.0, 20.0
    )

    expected_A = np.array([
        [1, 0, 0],
        [-2, 4, -2],
        [0, -4, 6],
    ])

    np.testing.assert_allclose(A, expected_A)
    np.testing.assert_allclose(b, [100, 0, 40])


def test_boundary_conditions():
    """Comprobar las condiciones Dirichlet y Robin."""
    x = create_mesh_1d(0, 1, 41)

    u = solve_diffusion_stationary_1d(
        x, 10.0, 0.0, 100.0, 20.0, 20.0
    )

    # Solucion analitica para k constante y q = 0
    exact = 100.0 - (160.0 / 3.0) * x

    np.testing.assert_allclose(u, exact, atol=1e-9)


def test_variable_conductivity():
    """Comparar con una solucion analitica de k variable."""
    x = create_mesh_1d(0, 1, 81)

    # k(x) = 1 + x, u(x) = ln(1 + x)
    exact = np.log1p(x)

    u = solve_diffusion_stationary_1d(
        x,
        lambda x: 1.0 + x,
        0.0,
        0.0,
        2.0,
        np.log(2.0) + 0.5,
    )

    error = np.max(np.abs(u - exact))

    assert error < 1e-5
