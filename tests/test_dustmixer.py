from synthesizer.dustmixer import bhcoat, bhmie
from synthesizer.dustmixer.dustmixer import DustMixer


def test_bhmie_scattering():
    qext, qsca, qback, g = bhmie.bhmie(1.0, 1.5 + 0.1j, 10)
    assert qext > 0
    assert qsca >= 0


def test_bhcoat_scattering():
    qext, qsca, qback, g = bhcoat.bhcoat(1.0, 1.2, 1.5 + 0.1j, 1.6 + 0.2j, 10)
    assert qext > 0


def test_dustmixer_init():
    dm = DustMixer(amin=0.1, amax=10.0, na=10)
    assert dm.amin == 0.1
    assert dm.amax == 10.0
    assert dm.na == 10
