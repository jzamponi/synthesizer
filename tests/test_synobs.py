from synthesizer.synobs.scripter import CasaScript


def test_casascript_init():
    # CasaScript(lam, name="casa_script.py")
    cs = CasaScript(lam=1300)
    assert cs.lam == 1300
    assert cs.name == "casa_script.py"
    assert cs.simobserve is True
    assert cs.clean is True


def test_casascript_custom_name():
    cs = CasaScript(lam=850, name="my_script.py")
    assert cs.lam == 850
    assert cs.name == "my_script.py"
    assert cs.freq > 0
