
"""Solvers for linear systems."""

import numpy as np


def solve_tridiagonal(A, b):
    """Solve a tridiagonal system using the Thomas algorithm."""

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    if A.ndim != 2 or b.ndim != 1 or A.shape != (b.size, b.size):
        raise ValueError("Invalid matrix or vector dimensions")

    if b.size < 2 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Invalid system data")

    if np.any(np.triu(A, 2) != 0) or np.any(np.tril(A, -2) != 0):
        raise ValueError("Matrix must be tridiagonal")

    # Extract diagonals
    lower = np.diag(A, -1).copy()
    main = np.diag(A).copy()
    upper = np.diag(A, 1).copy()
    rhs = b.copy()

    n = len(b)

    # Forward elimination
    for i in range(1, n):
        if main[i - 1] == 0:
            raise ValueError("Zero pivot")

        factor = lower[i - 1] / main[i - 1]

        main[i] -= factor * upper[i - 1]
        rhs[i] -= factor * rhs[i - 1]

    if main[-1] == 0:
        raise ValueError("Zero pivot")

    # Back substitution
    u = np.zeros(n)
    u[-1] = rhs[-1] / main[-1]

    for i in range(n - 2, -1, -1):
        u[i] = (rhs[i] - upper[i] * u[i + 1]) / main[i]

    return u
