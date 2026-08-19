from pathlib import Path
from synthesizer import utils


def test_constants():
    assert utils.au > 0
    assert utils.pc > utils.au
    assert utils.msun > 0
    assert utils.rsun > 0
    assert utils.lsun > 0
    assert utils.grav > 0
    assert utils.kb > 0
    assert utils.c_light > 0


def test_basename():
    assert utils.basename("/path/to/file.fits") == "file"
    assert utils.basename("/path/to/file.fits", dot=False) == "file.fits"
    assert utils.basename("simple_name") == "simple_name"


def test_color_class():
    assert utils.color.red.startswith("\033")
    assert utils.color.none == "\033[0m"


def test_which_function():
    # python3 should exist in PATH
    assert utils.which("python3") is True
    # non-existent command should return False
    assert utils.which("non_existent_command_12345") is False
