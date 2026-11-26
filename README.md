# MiniPDE

Proyecto transversal de la asignatura **Métodos Numéricos en Ecuaciones
en Derivadas Parciales**.

## Objetivo

Desarrollar progresivamente un pequeño solver modular para ecuaciones
en derivadas parciales, aplicando:

- discretización espacial;
- condiciones de contorno;
- resolución de sistemas lineales y no lineales;
- integración temporal;
- análisis de estabilidad;
- verificación numérica;
- pruebas automatizadas;
- control de versiones con Git.

## Evolución prevista

| Versión | Contenido |
|---|---|
| v0.1 | Entorno, estructura, punto flotante y pruebas |
| v0.2 | Operadores de diferencias finitas |
| v0.3 | Ecuación de Poisson y condiciones de contorno |
| v0.4 | Solvers lineales y no lineales |
| v0.5 | Integradores temporales |
| v0.6 | Ecuación del calor |
| v0.7 | Ecuación de advección |
| v1.0 | Verificación, documentación y proyecto final |

## Instalación

### Crear el entorno virtual

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

## Actividad 1. Difusión estacionaria con conductividad variable

### Descripción

En esta actividad he desarrollado un programa para resolver un problema de difusión térmica estacionaria en una dimensión, considerando una conductividad variable, generación interna de calor y condiciones de contorno de Dirichlet y Robin.

Para resolverlo he utilizado diferencias finitas conservativas, calculando los flujos en las interfaces y utilizando la media aritmética para aproximar la conductividad.

He trabajado a partir de la primera versión de MiniPDE, añadiendo progresivamente los archivos y funciones que hemos ido viendo en clase, en lugar de partir directamente de la versión 4.

### Archivos principales

- `src/minipde/models/diffusion_stationary.py`: construcción del sistema de ecuaciones y resolución mediante NumPy.
- `src/minipde/linear_solvers.py`: implementación del algoritmo de Thomas para resolver sistemas tridiagonales.
- `examples/activity1_diffusion.py`: comprobación del método mediante una solución manufacturada y estudio de convergencia.
- `examples/activity1_physical_case.py`: aplicación del método a un caso físico de conducción de calor.
- `tests/test_diffusion_stationary.py`: pruebas del método, condiciones de contorno, convergencia y balance energético.
- `tests/test_linear_solvers.py`: pruebas del algoritmo de Thomas.

### Casos estudiados

**1. Solución manufacturada**

He utilizado una solución analítica conocida para comprobar el funcionamiento del método. Para ello, he trabajado con cuatro mallas de 21, 41, 81 y 161 nodos.

Los resultados muestran una convergencia de segundo orden, tanto para el error máximo como para el error L2.

**2. Caso físico**

He considerado una barra de 1 m de longitud con las siguientes condiciones:

- Conductividad: k(x) = 10(1 + x) W/(m·K).
- Generación interna de calor: 100 W/m³.
- Temperatura en el extremo izquierdo: 100 °C.
- Coeficiente de convección: 20 W/(m²·K).
- Temperatura ambiente: 20 °C.

Con la malla de 161 nodos, la temperatura obtenida en el extremo derecho es aproximadamente 55,1436 °C.

### Comprobaciones realizadas

Para comprobar que los resultados eran correctos, he realizado las siguientes verificaciones:

- Estudio de convergencia con cuatro mallas.
- Cálculo del residuo algebraico.
- Comprobación del balance global de energía.
- Comparación de las soluciones obtenidas con NumPy y el algoritmo de Thomas.
- Comparación de los resultados del caso físico al refinar la malla.

Los residuos, los defectos del balance y las diferencias entre los dos solvers son muy pequeños, del orden de la precisión numérica. Además, los 18 tests automatizados pasan correctamente.

### Cómo ejecutar los programas

Desde la carpeta principal del repositorio, con el entorno de Python preparado:

**Instalar el proyecto y sus dependencias:**

`python -m pip install -e ".[dev]"`

**Ejecutar el estudio de convergencia:**

`python examples/activity1_diffusion.py`

**Ejecutar el caso físico:**

`python examples/activity1_physical_case.py`

**Ejecutar los tests:**

`python -m pytest -q`

**Comprobar el código con Ruff:**

`python -m ruff check .`

### Resultados

Las gráficas generadas se encuentran en la carpeta `figures`:

- `activity1_convergence.png`: evolución del error al refinar la malla.
- `activity1_physical_temperature.png`: distribución de temperatura a lo largo de la barra.

También se muestran por terminal las tablas de convergencia, los residuos, los balances energéticos y la comparación entre solvers.

### GitHub

El desarrollo de esta actividad se encuentra en la rama `feature/activity1-diffusion`.

https://github.com/elenalcalde/minipde-starter/tree/feature/activity1-diffusion