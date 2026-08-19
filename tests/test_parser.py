from synthesizer.parser import synthesizer_parser


def test_parser_defaults():
    parser = synthesizer_parser()
    args = parser.parse_args([])
    assert args.grid is False
    assert args.raytrace is False
    assert args.model == "ppdisk"
    assert args.ncells == 100
    assert args.nphot == 1e5
    assert args.lam == 1300


def test_parser_custom_args():
    parser = synthesizer_parser()
    args = parser.parse_args([
        "--grid",
        "--model", "powerlaw",
        "--ncells", "50",
        "--lam", "3000",
        "--overwrite"
    ])
    assert args.grid is True
    assert args.model == "powerlaw"
    assert args.ncells == 50
    assert args.lam == 3000
    assert args.overwrite is True
