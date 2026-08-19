from synthesizer.raytrace.radmc3d import RADMC3D


def test_radmc3d_init():
    rt = RADMC3D(mode="image")
    assert rt.mode == "image"
    assert rt.logfile == "radmc3d.out"
    assert rt.fitsfile == "radmc3d_I.fits"


def test_radmc3d_set_command():
    rt = RADMC3D(mode="image")
    rt.set_command(
        lam=1300, sizeau=400, incl=40, npix=200, noscat=False, stokes="I", extra=""
    )
    assert rt.lam == 1300
    assert rt.npix == 200
