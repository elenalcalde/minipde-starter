
"""Estudio de convergencia de diferencias finitas."""

import numpy as np

from minipde.mesh import create_mesh_1d
from minipde.operators import first_derivative, second_derivative
from minipde.verification import max_error, observed_order

# Numero de nodos para cada refinamiento
N_values = [21, 41, 81, 161, 321]

h_values = []
error_first = []
error_second = []

print(f"{'N':>5} {'h':>12} {'Error 1a':>15} {'Error 2a':>15}")

for N in N_values:

    # Generar malla equiespaciada
    x = create_mesh_1d(0, 2 * np.pi, N)
    h = x[1] - x[0]

    # Funcion de referencia
    u = np.sin(x)

    # Derivadas numericas
    du_num = first_derivative(x, u)
    d2u_num = second_derivative(x, u)

    # Derivadas exactas
    du_exact = np.cos(x)
    d2u_exact = -np.sin(x)

    # Errores maximos
    e1 = max_error(du_num, du_exact)
    e2 = max_error(d2u_num, d2u_exact)

    h_values.append(h)
    error_first.append(e1)
    error_second.append(e2)

    print(f"{N:5d} {h:12.5e} {e1:15.5e} {e2:15.5e}")


# Orden de convergencia observado
p_first = observed_order(h_values, error_first)
p_second = observed_order(h_values, error_second)

print("\nOrden observado - Primera derivada:")
print(p_first)

print("\nOrden observado - Segunda derivada:")
print(p_second)
