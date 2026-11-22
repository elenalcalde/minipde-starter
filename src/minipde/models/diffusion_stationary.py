
"""Difusión estacionaria 1D con conductividad variable."""

import numpy as np


def _nodal_values(data, x):
    """Evaluar una función, escalar o array en los nodos."""
    values = data(x) if callable(data) else data
    values = np.asarray(values, dtype=float)

    if values.ndim == 0:
        values = np.full(x.shape, float(values))

    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError("Datos nodales no válidos")

    return values


def assemble_diffusion_stationary_1d(
    x, conductivity, source, u_left, h_conv, u_inf
):
    """Construir el sistema A u = b mediante flujos conservativos."""

    x = np.asarray(x, dtype=float)

    # Validar la malla
    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("La malla debe tener al menos 3 nodos")

    dx = x[1] - x[0]

    if dx <= 0 or not np.allclose(
        np.diff(x), dx, rtol=1e-10, atol=0.0
    ):
        raise ValueError("La malla debe ser uniforme y creciente")

    # Conductividad y fuente
    k = _nodal_values(conductivity, x)
    q = _nodal_values(source, x)

    if np.any(k <= 0):
        raise ValueError("La conductividad debe ser positiva")

    if not np.all(np.isfinite([u_left, h_conv, u_inf])):
        raise ValueError("Parámetros de contorno no válidos")

    if h_conv <= 0:
        raise ValueError("h_conv debe ser positivo")

    n = len(x)

    # Conductividad en interfaces: media aritmética
    k_half = 0.5 * (k[:-1] + k[1:])

    # Sistema lineal
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Condición de Dirichlet: extremo izquierdo
    A[0, 0] = 1.0
    b[0] = u_left

    # Nodos interiores: balance conservativo
    for i in range(1, n - 1):
        k_w = k_half[i - 1]
        k_e = k_half[i]

        A[i, i - 1] = -k_w
        A[i, i] = k_w + k_e
        A[i, i + 1] = -k_e

        b[i] = q[i] * dx**2

    # Condición de Robin: extremo derecho
    # Medio volumen de control
    A[-1, -2] = -k_half[-1] / dx
    A[-1, -1] = k_half[-1] / dx + h_conv

    b[-1] = h_conv * u_inf + q[-1] * dx / 2.0

    return A, b


def solve_diffusion_stationary_1d(
    x, conductivity, source, u_left, h_conv, u_inf
):
    """Resolver el sistema mediante NumPy."""

    A, b = assemble_diffusion_stationary_1d(
        x, conductivity, source, u_left, h_conv, u_inf
    )

    return np.linalg.solve(A, b)
