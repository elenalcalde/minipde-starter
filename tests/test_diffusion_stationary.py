
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


def test_second_order_convergence():
    """Comprobar convergencia de segundo orden."""
    errors = []

    for n in [21, 41, 81, 161]:
        x = create_mesh_1d(0, 1, n)

        exact = 100 - 20 * x - 10 * x**2

        u = solve_diffusion_stationary_1d(
            x,
            lambda x: 10 * (1 + x),
            lambda x: 400 * (1 + x),
            100,
            20,
            30,
        )

        errors.append(np.max(np.abs(u - exact)))

    orders = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))

    assert np.allclose(orders, 2.0, atol=0.01)


def test_global_energy_balance():
    """Comprobar conservacion global de energia."""
    x = create_mesh_1d(0, 1, 81)
    dx = x[1] - x[0]

    A, b = assemble_diffusion_stationary_1d(
        x, lambda x: 10 * (1 + x), 100, 100, 20, 20
    )

    u = np.linalg.solve(A, b)

    k_half = 0.5 * (10 * (1 + x[0]) + 10 * (1 + x[1]))

    flux_in = -k_half * (u[1] - u[0]) / dx - 100 * dx / 2
    flux_out = 20 * (u[-1] - 20)

    balance = flux_out - flux_in - 100

    assert abs(balance) < 1e-8
    assert np.linalg.norm(A @ u - b, ord=np.inf) < 1e-8
