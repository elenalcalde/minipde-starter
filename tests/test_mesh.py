
"""Tests para la generación de mallas 1D."""

import numpy as np
import pytest

from minipde.mesh import create_mesh_1d


def test_mesh_spacing():
    """Comprobar extremos, número de nodos y espaciamiento."""
    x = create_mesh_1d(0.0, 1.0, 6)

    assert len(x) == 6
    assert x[0] == 0.0
    assert x[-1] == 1.0
    np.testing.assert_allclose(np.diff(x), 0.2)


def test_mesh_invalid_points():
    """Comprobar que no se admite una malla con un solo nodo."""
    with pytest.raises(ValueError):
        create_mesh_1d(0.0, 1.0, 1)
