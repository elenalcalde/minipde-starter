
import numpy as np


def create_mesh_1d(x_min: float, x_max: float, N: int) -> np.ndarray:
    """
    Genera una malla 1D equiespaciada.

    Parameters
    ----------
    x_min : float
        Extremo izquierdo del dominio.
    x_max : float
        Extremo derecho del dominio.
    N : int
        Numero de puntos de la malla.

    Returns
    -------
    np.ndarray
        Coordenadas de los N nodos.
    """

    # Comprobacion de los argumentos
    if not isinstance(N, (int, np.integer)) or isinstance(N, (bool, np.bool_)) or N < 2:
        raise ValueError("N debe ser un entero mayor o igual que 2.")

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError("El extremo derecho debe ser mayor que el izquierdo.")

    # Generacion de la malla equiespaciada
    x = np.linspace(x_min, x_max, N)

    return x
