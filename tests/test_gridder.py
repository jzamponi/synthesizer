import numpy as np

from synthesizer.gridder.models import Constant, PPdisk
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
    # PPdisk(x, y, z, field, rin, rout, rc, r0, h0, alpha, flare, mdisk, mstar, mdot)
    x = np.ones((5, 5, 5))
    y = np.ones((5, 5, 5))
    z = np.ones((5, 5, 5))
    disk = PPdisk(
        x,
        y,
        z,
        field="z",
        rin=1,
        rout=100,
        rc=140,
        r0=100,
        h0=5,
        alpha=1,
        flare=0.5,
        mdisk=1e-3,
        mstar=1,
        mdot=1e-5,
    )
    assert disk is not None
    assert disk.rin == 1
    assert disk.rout == 100
