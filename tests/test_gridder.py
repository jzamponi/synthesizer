import numpy as np
from synthesizer.gridder.analytical import PPDisk
from synthesizer.gridder.models import Constant, PowerLaw
from synthesizer.gridder.vector_field import VectorField


def test_vector_field():
    x = np.linspace(-10, 10, 5)
    y = np.linspace(-10, 10, 5)
    z = np.linspace(-10, 10, 5)
    vf = VectorField(x, y, z, morphology="z")
    assert vf is not None


def test_constant_model():
    x = np.ones((5, 5, 5))
    y = np.ones((5, 5, 5))
    z = np.ones((5, 5, 5))
    c = Constant(x, y, z, field="z", rho0=1e-12)
    assert c.dens.shape == (5, 5, 5)
    assert np.all(c.dens == 1e-12)


def test_ppdisk_model():
    disk = PPDisk(mdisk=1e-3, rin=1, rout=100)
    assert disk.mdisk == 1e-3
    assert disk.rin == 1
    assert disk.rout == 100
