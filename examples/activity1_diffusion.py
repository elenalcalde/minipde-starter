
"""Actividad 1: estudio de convergencia."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from minipde.linear_solvers import solve_tridiagonal
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


# ============================================================
# TABLA DE RESIDUOS Y BALANCE ENERGETICO - 4 MALLAS
# ============================================================
verification_rows = []
print("\nVERIFICACION NUMERICA - CASO MANUFACTURADO")
print(
    f"{'N':>5} {'Residuo':>15} "
    f"{'Defecto balance':>20} {'Diferencia solvers':>20}"
)
print("-" * 65)

for n in [21, 41, 81, 161]:
    x_test = create_mesh_1d(0, 1, n)
    dx_test = x_test[1] - x_test[0]

    # Construccion del sistema
    A_test, b_test = assemble_diffusion_stationary_1d(
        x_test,
        conductivity=lambda x: 10 * (1 + x),
        source=lambda x: 400 * (1 + x),
        u_left=100,
        h_conv=20,
        u_inf=30,
    )

    # Comparacion entre solvers
    u_numpy = np.linalg.solve(A_test, b_test)
    u_thomas = solve_tridiagonal(A_test, b_test)

    difference = np.max(np.abs(u_numpy - u_thomas))

    # Residuo algebraico
    residual = np.linalg.norm(A_test @ u_numpy - b_test, ord=np.inf)

    # Flujo de entrada
    k_half = 0.5 * (
        10 * (1 + x_test[0]) + 10 * (1 + x_test[1])
    )

    q_values = 400 * (1 + x_test)

    flux_half = -k_half * (u_numpy[1] - u_numpy[0]) / dx_test

    flux_in = flux_half - q_values[0] * dx_test / 2

    # Flujo de salida (Robin)
    flux_out = 20 * (u_numpy[-1] - 30)

    # Generacion termica total
    heat_generated = np.trapezoid(q_values, x_test)

    # Defecto del balance
    balance = flux_out - flux_in - heat_generated

    print(
        f"{n:5d} {residual:15.6e} "
        f"{balance:20.6e} {difference:20.6e}"
    )

    verification_rows.append([n, residual, balance, difference])

# ============================================================
# COMPARACION ENTRE SOLUCION EXACTA Y NUMERICA
# ============================================================

# Malla de 161 nodos
x_plot = create_mesh_1d(0, 1, 161)

# Solucion exacta manufacturada
u_exact = 100 - 20 * x_plot - 10 * x_plot**2

# Solucion numerica
u_num = solve_diffusion_stationary_1d(
    x_plot,
    conductivity=lambda x: 10 * (1 + x),
    source=lambda x: 400 * (1 + x),
    u_left=100,
    h_conv=20,
    u_inf=30,
)

# Representacion grafica
plt.figure(figsize=(8, 5))

plt.plot(x_plot, u_exact, "k-", linewidth=2,
         label="Solucion exacta")

plt.plot(x_plot, u_num, "r--", linewidth=2,
         label="Solucion numerica (N=161)")

plt.xlabel("Posicion x (m)")
plt.ylabel("Temperatura (°C)")
plt.title("Comparacion entre solucion exacta y numerica")

plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()

# Guardar figura
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/activity1_exact_vs_numerical.png", dpi=300)

plt.show()


# ============================================================
# GUARDAR TABLA DE CONVERGENCIA EN CSV
# ============================================================

# Calcular el orden observado entre mallas
orders_inf = [np.nan]
orders_l2 = [np.nan]

for i in range(1, len(mesh_sizes)):
    ratio_h = h_values[i - 1] / h_values[i]

    orders_inf.append(
        np.log(linf_values[i - 1] / linf_values[i]) / np.log(ratio_h)
    )

    orders_l2.append(
        np.log(l2_values[i - 1] / l2_values[i]) / np.log(ratio_h)
    )

# Crear carpeta para los resultados
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Preparar tabla
convergence_table = np.column_stack((
    mesh_sizes,
    h_values,
    linf_values,
    l2_values,
    orders_inf,
    orders_l2,
))

# Guardar archivo CSV
np.savetxt(
    results_dir / "activity1_convergence.csv",
    convergence_table,
    delimiter=",",
    header="N,dx,error_Linf,error_L2,p_Linf,p_L2",
    comments="",
    fmt=["%d", "%.8e", "%.8e", "%.8e", "%.4f", "%.4f"],
)

print("\nTabla de convergencia guardada en results/activity1_convergence.csv")


# ============================================================
# GUARDAR TABLA DE RESIDUOS Y BALANCES EN CSV
# ============================================================

verification_table = np.array(verification_rows)

np.savetxt(
    results_dir / "activity1_verification.csv",
    verification_table,
    delimiter=",",
    header="N,residuo,defecto_balance,diferencia_solvers",
    comments="",
    fmt=["%d", "%.8e", "%.8e", "%.8e"],
)

print("Tabla de verificacion guardada en results/activity1_verification.csv")
