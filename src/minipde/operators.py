"""Aproximaciones de derivadas en una malla 1D equiespaciada.

Las funciones reciben dos arrays de la misma longitud:
- x: coordenadas de los nodos (ordenadas y equiespaciadas).
- u: valores de la funcion u(x) en esos nodos.

Devuelven un array con la derivada aproximada en cada nodo.
Se utilizan diferencias centradas en el interior y diferencias
unilaterales de segundo orden en los extremos.
"""

import numpy as np


def _validate_mesh_and_values(x: np.ndarray, u: np.ndarray, min_points: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Comprueba los argumentos y devuelve los arrays y el paso de malla."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)

    if x.ndim != 1 or u.ndim != 1:
        raise ValueError("x y u deben ser arrays unidimensionales.")
    if len(x) != len(u):
        raise ValueError("x y u deben tener el mismo numero de elementos.")
    if len(x) < min_points:
        raise ValueError(f"Se necesitan al menos {min_points} nodos.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)):
        raise ValueError("x y u deben contener solo valores finitos.")

    steps = np.diff(x)
    if np.any(steps <= 0):
        raise ValueError("Los nodos x deben estar ordenados de menor a mayor.")
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=0.0):
        raise ValueError("La malla x debe ser equiespaciada.")

    return x, u, float(steps[0])


def first_derivative(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Aproxima du/dx en todos los nodos con error de orden O(h**2).

    Interior: diferencia centrada.
    Extremos: diferencia unilateral de segundo orden.
    Requiere al menos 3 nodos.
    """
    x, u, h = _validate_mesh_and_values(x, u, min_points=3)
    dudx = np.empty_like(u)

    dudx[1:-1] = (u[2:] - u[:-2]) / (2.0 * h)
    dudx[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
    dudx[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * h)

    return dudx


def second_derivative(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Aproxima d2u/dx2 en todos los nodos con error de orden O(h**2).

    Interior: diferencia centrada.
    Extremos: diferencia unilateral de segundo orden.
    Requiere al menos 4 nodos.
    """
    x, u, h = _validate_mesh_and_values(x, u, min_points=4)
    d2udx2 = np.empty_like(u)

    d2udx2[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / h**2
    d2udx2[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - u[3]) / h**2
    d2udx2[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - u[-4]) / h**2

    return d2udx2
