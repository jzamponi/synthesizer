from synthesizer.synobs.scripter import CasaScript


def test_casascript_init():
    cs = CasaScript(telescope="ALMA", lam=1300)
    assert cs.telescope == "ALMA"
    assert cs.lam == 1300
    assert cs.scriptfile == "casa_script.py"
