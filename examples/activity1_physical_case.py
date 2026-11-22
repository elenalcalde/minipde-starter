
"""Actividad 1: caso fisico de difusion estacionaria."""

from pathlib import Path

import matplotlib.pyplot as plt

from minipde.mesh import create_mesh_1d
from minipde.models.diffusion_stationary import (
    solve_diffusion_stationary_1d,
)

# Parametros fisicos
L = 1.0
N = 161
q = 100.0
u_left = 100.0
h_conv = 20.0
u_inf = 20.0

# Malla y conductividad variable
x = create_mesh_1d(0, L, N)
dx = x[1] - x[0]

def conductivity(x):
    return 10 * (1 + x)

# Resolver
u = solve_diffusion_stationary_1d(
    x, conductivity, q, u_left, h_conv, u_inf
)

# Balance energetico
k_half = 0.5 * (conductivity(x[0]) + conductivity(x[1]))

flux_in = -k_half * (u[1] - u[0]) / dx - q * dx / 2
flux_out = h_conv * (u[-1] - u_inf)
heat_generated = q * L

balance = flux_out - flux_in - heat_generated

print("CASO FISICO - N = 161")
print(f"Temperatura izquierda: {u[0]:.6f} °C")
print(f"Temperatura derecha:   {u[-1]:.6f} °C")
print(f"Flujo de entrada:      {flux_in:.6f} W/m²")
print(f"Flujo de salida:       {flux_out:.6f} W/m²")
print(f"Calor generado:        {heat_generated:.6f} W/m²")
print(f"Defecto del balance:   {balance:.6e} W/m²")

# Grafica de temperatura
plt.figure(figsize=(8, 5))
plt.plot(x, u, linewidth=2, label="Temperatura numerica")

plt.xlabel("Posicion x (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Difusion estacionaria - Caso fisico")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/activity1_physical_temperature.png", dpi=300)
plt.show()


# Comparacion de mallas para el caso fisico
print("\nCOMPARACION DE MALLAS - CASO FISICO")
print(" N       dx         T_final (°C)     Variacion (°C)")
print("-" * 58)

previous_temperature = None

for n in [21, 41, 81, 161]:
    x_test = create_mesh_1d(0, L, n)
    dx_test = x_test[1] - x_test[0]

    u_test = solve_diffusion_stationary_1d(
        x_test, conductivity, q, u_left, h_conv, u_inf
    )

    final_temperature = u_test[-1]

    if previous_temperature is None:
        variation = float("nan")
    else:
        variation = abs(final_temperature - previous_temperature)

    print(
        f"{n:3d}   {dx_test:.6f}     "
        f"{final_temperature:.6f}        {variation:.6e}"
    )

    previous_temperature = final_temperature
