
"""Actividad 1: estudio de convergencia."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minipde.mesh import create_mesh_1d
from minipde.models.diffusion_stationary import (
    assemble_diffusion_stationary_1d,
    solve_diffusion_stationary_1d,
)

mesh_sizes = [21, 41, 81, 161]

previous_h = None
previous_linf = None
previous_l2 = None

print(" N      dx         L_inf         L2       p_inf    p_L2")
print("-" * 65)
h_values = []
linf_values = []
l2_values = []

for n in mesh_sizes:
    x = create_mesh_1d(0, 1, n)
    dx = x[1] - x[0]

    # Solucion manufacturada
    exact = 100 - 20 * x - 10 * x**2

    # Solucion numerica
    u = solve_diffusion_stationary_1d(
        x,
        conductivity=lambda x: 10 * (1 + x),
        source=lambda x: 400 * (1 + x),
        u_left=100,
        h_conv=20,
        u_inf=30,
    )

    # Errores
    error = u - exact
    linf = np.max(np.abs(error))

    # Norma L2 con integracion trapezoidal
    l2 = np.sqrt(
        dx * (
            0.5 * error[0]**2
            + np.sum(error[1:-1]**2)
            + 0.5 * error[-1]**2
        )
    )

    # Orden observado
    if previous_h is None:
        p_inf = float("nan")
        p_l2 = float("nan")
    else:
        p_inf = np.log(previous_linf / linf) / np.log(previous_h / dx)
        p_l2 = np.log(previous_l2 / l2) / np.log(previous_h / dx)

    print(
        f"{n:3d}  {dx:.6f}  {linf:.6e}  "
        f"{l2:.6e}  {p_inf:6.3f}  {p_l2:6.3f}"
    )
    h_values.append(dx)
    linf_values.append(linf)
    l2_values.append(l2)
    previous_h = dx
    previous_linf = linf
    previous_l2 = l2

# Grafica de convergencia
h_values = np.array(h_values)

reference = linf_values[0] * (h_values / h_values[0])**2

plt.figure(figsize=(8, 5))

plt.loglog(h_values, linf_values, "o-", label="Error L_inf")
plt.loglog(h_values, l2_values, "s-", label="Error L2")
plt.loglog(h_values, reference, "k--", label="Referencia O(dx^2)")

plt.xlabel("Espaciamiento de malla (dx)")
plt.ylabel("Error")
plt.title("Convergencia - Difusion estacionaria 1D")

plt.grid(True, which="both", linestyle=":")
plt.legend()
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/activity1_convergence.png", dpi=300)

plt.show()


# Verificacion algebraica y balance energetico
# Utilizamos la ultima malla: N = 161

A, b = assemble_diffusion_stationary_1d(
    x,
    conductivity=lambda x: 10 * (1 + x),
    source=lambda x: 400 * (1 + x),
    u_left=100,
    h_conv=20,
    u_inf=30,
)

# Residuo algebraico
residual = np.linalg.norm(A @ u - b, ord=np.inf)

# Flujo de salida: condicion Robin
flux_out = 20 * (u[-1] - 30)

# Flujo de entrada reconstruido desde la primera interfaz
k_half_left = 0.5 * (10 * (1 + x[0]) + 10 * (1 + x[1]))
flux_half_left = -k_half_left * (u[1] - u[0]) / dx
flux_in = flux_half_left - 0.5 * 400 * dx

# Calor generado: regla trapezoidal
q = 400 * (1 + x)
heat_generated = np.trapezoid(q, x)

# Defecto del balance global
balance = flux_out - flux_in - heat_generated

print("\nVERIFICACION FINAL - N = 161")
print(f"Residuo algebraico: {residual:.6e}")
print(f"Flujo de entrada:   {flux_in:.6f}")
print(f"Flujo de salida:    {flux_out:.6f}")
print(f"Calor generado:     {heat_generated:.6f}")
print(f"Defecto del balance:{balance:.6e}")
