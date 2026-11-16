
"""Estudio de convergencia del problema de Poisson 1D."""

import numpy as np

from minipde.mesh import create_mesh_1d
from minipde.poisson import assemble_poisson_1d, solve_poisson_1d
from minipde.verification import max_error, observed_order

N_values = [21, 41, 81, 161]

h_values = []
errors = []

print(f"{'N':>5} {'h':>12} {'Error maximo':>15} {'Residuo':>15}")

for N in N_values:
    # Malla equiespaciada
    x = create_mesh_1d(0, 1, N)
    h = x[1] - x[0]

    # Solucion exacta y termino fuente
    u_exact = np.sin(np.pi * x)
    source = lambda x: np.pi**2 * np.sin(np.pi * x)

    # Solucion numerica
    u_num = solve_poisson_1d(x, source)

    # Error y residuo algebraico
    error = max_error(u_num, u_exact)

    A, b = assemble_poisson_1d(x, source)
    residual = np.linalg.norm(A @ u_num - b, ord=np.inf)

    h_values.append(h)
    errors.append(error)

    print(f"{N:5d} {h:12.5e} {error:15.5e} {residual:15.5e}")

# Orden de convergencia
orders = observed_order(h_values, errors)

print("\nOrden de convergencia observado:")
print(orders)
