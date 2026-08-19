from synthesizer.pipeline import Pipeline


def test_pipeline_init_defaults():
    p = Pipeline(nthreads=2, overwrite=True)
    assert p.nthreads == 2
    assert p.overwrite is True
    assert p.distance == 140
    assert p.lam == 1300


def test_pipeline_create_grid_ppdisk():
    p = Pipeline(nthreads=1, overwrite=True)
    p.create_grid(
        model="ppdisk",
        mdisk=1e-3,
        rin=1,
        rout=100,
        ncells=10,
        bbox=100,
        temperature=True,
    )
    assert p.grid is not None
    assert hasattr(p.grid, "dens")
    assert hasattr(p.grid, "temp")
