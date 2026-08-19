import pytest

from synthesizer import utils


def test_color_class():
    assert utils.color.red.startswith("\033")
    assert utils.color.none == "\033[0m"
    assert utils.color.blue.startswith("\033")


def test_basename_with_path():
    assert utils.basename("/path/to/file.fits") == "file"


def test_basename_without_dot():
    assert utils.basename("/path/to/file.fits", dot=False) == "file.fits"


def test_basename_no_slash():
    assert utils.basename("simple_name") == "simple_name"


def test_which_found():
    # python3 should exist, no exception raised
    utils.which("python3")


def test_which_not_found():
    with pytest.raises(utils.NotInstalled):
        utils.which("non_existent_command_12345")


def test_print_function():
    # Should not raise
    utils.print_("test message", verbose=True)
    utils.print_("test message", verbose=False)
