import numpy as np

from synthesizer.dustmixer import bhcoat, bhmie
from synthesizer.dustmixer.dustmixer import Dust


def test_bhmie_scattering():
    # bhmie expects theta as numpy array of angles 0-180 degrees
    theta = np.linspace(0, 180, 10)
    S1, S2, qext, qsca, qabs, qback, gsca = bhmie.bhmie(1.0, 1.5 + 0.1j, theta)
    assert qext > 0
    assert qsca >= 0
    assert qabs >= 0
    assert len(S1) == len(theta)


def test_bhcoat_scattering():
    # bhcoat(x, y, rfrel1, rfrel2) -> (qqext, qqsca, qqabs, qback)
    qext, qsca, qabs, qback = bhcoat.bhcoat(1.0, 1.2, 1.5 + 0.1j, 1.6 + 0.2j)
    assert qext > 0


def test_dust_init():
    # Dust(name, n, k, dens, l, scatmatrix, pb)
    dm = Dust(name="silicate", n=1.5, k=0.01, dens=3.0)
    assert dm.name == "silicate"
    assert dm.dens == 3.0
    assert dm.nlam > 0
