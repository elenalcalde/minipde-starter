
"""Solver for the one-dimensional Poisson equation."""

import numpy as np


def assemble_poisson_1d(x, source, left=0.0, right=0.0):
    """Assemble the linear system A u = b for -u'' = f."""

    x = np.asarray(x, dtype=float)

    if x.ndim != 1 or x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError("x must contain at least three finite points")

    h = x[1] - x[0]

    if h <= 0 or not np.allclose(np.diff(x), h):
        raise ValueError("The mesh must be uniform and increasing")

    f = source(x) if callable(source) else source
    f = np.asarray(f, dtype=float)

    if f.shape != x.shape or not np.all(np.isfinite(f)):
        raise ValueError("Invalid source values")

    if not np.all(np.isfinite([left, right])):
        raise ValueError("Boundary values must be finite")

    n = len(x)

    A = np.zeros((n, n))
    b = f.copy()

    # Interior nodes: centered finite differences
    for i in range(1, n - 1):
        A[i, i - 1] = -1.0 / h**2
        A[i, i] = 2.0 / h**2
        A[i, i + 1] = -1.0 / h**2

    # Dirichlet boundary conditions
    A[0, 0] = 1.0
    b[0] = left

    A[-1, -1] = 1.0
    b[-1] = right

    return A, b


def solve_poisson_1d(x, source, left=0.0, right=0.0):
    """Solve the one-dimensional Poisson equation."""

    A, b = assemble_poisson_1d(x, source, left, right)

    return np.linalg.solve(A, b)
