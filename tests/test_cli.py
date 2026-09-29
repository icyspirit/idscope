import pytest

from idscope.cli import main
from idscope.demo import DEMO_CONFIG


def test_list_ids(data_dir, capsys):
    assert main(["--list", str(data_dir / "a.nc")]) == 0
    lines = capsys.readouterr().out.split()
    assert {"core_profiles", "equilibrium", "magnetics", "summary"} <= set(lines)


def test_list_quantities(data_dir, capsys):
    assert main(["--list", str(data_dir / "a.nc"), "core_profiles"]) == 0
    out = capsys.readouterr().out
    assert "profile   core_profiles/profiles_1d/electrons/temperature" in out
    assert "[eV]" in out


def test_list_templates(capsys):
    assert main(["--list-templates"]) == 0
    assert capsys.readouterr().out.split() == ["core_profiles", "equilibrium", "pf_active"]


def test_template_export(tmp_path, data_dir, capsys):
    pytest.importorskip("pyqtgraph.Qt")
    image = tmp_path / "eq.png"
    args = ["-T", "equilibrium", str(data_dir / "a.nc"), "-e", str(image), "-t", "5"]
    assert main(args) == 0
    assert image.stat().st_size > 10000
    assert "idscope:" not in capsys.readouterr().err  # no signal errors
    assert main(["-T", "equilibrium", "x.yaml"]) == 1
    assert main(["-T", "nope", str(data_dir / "a.nc")]) == 1


def test_errors(tmp_path, capsys):
    assert main([str(tmp_path / "missing.yaml")]) == 1
    assert "Cannot read" in capsys.readouterr().err
    assert main(["a.yaml", "b.yaml", "--export", "x.png"]) == 1
    assert main(["--list"]) == 1


def test_export(tmp_path, data_dir, capsys):
    pytest.importorskip("pyqtgraph.Qt")
    config = tmp_path / "demo.yaml"
    config.write_text(DEMO_CONFIG)
    image = tmp_path / "demo.png"
    args = [str(config), "-s", f"A={data_dir / 'a.nc'}", "-s", f"B={data_dir / 'h5'}",
            "-t", "2", "-e", str(image), "--size", "900x600"]
    assert main(args) == 0
    assert image.stat().st_size > 10000
    assert "wrote" in capsys.readouterr().out


def test_last_session(tmp_path, data_dir):
    from idscope.cli import assemble_config, build_parser
    from idscope.config import default_config, load_config
    from idscope.session import save_session

    parse = build_parser().parse_args
    assert assemble_config(parse([])) == (default_config(), None, False)  # no session yet

    scope = tmp_path / "demo.yaml"
    scope.write_text(DEMO_CONFIG)
    last = load_config(scope)
    last.time = 4.0
    save_session(last, scope, modified=True)
    assert assemble_config(parse([])) == (last, scope, True)

    # sources only: the panels of the last session, with other data
    config, path, modified = assemble_config(parse([str(data_dir / "a.nc"), "-t", "2"]))
    assert config.columns == last.columns
    assert [s.label for s in config.sources] == ["a"]
    assert config.time == 2.0
    assert (path, modified) == (scope, True)

    # a configuration file, a template or --new: not the last session
    assert assemble_config(parse([str(scope)]))[1:] == (scope, False)
    assert assemble_config(parse(["-T", "equilibrium"]))[1:] == (None, False)
    config, path, modified = assemble_config(parse(["--new", str(data_dir / "a.nc")]))
    assert config.columns == default_config().columns
    assert (path, modified) == (None, False)


def test_unreadable_session(capsys):
    from idscope.cli import assemble_config, build_parser
    from idscope.config import default_config
    from idscope.session import session_path

    session_path().parent.mkdir(parents=True)
    session_path().write_text("columns: [\n")
    assert assemble_config(build_parser().parse_args([])) == (default_config(), None, False)
    assert "Cannot read the last session" in capsys.readouterr().err


def test_export_with_the_last_session(tmp_path, data_dir):
    pytest.importorskip("pyqtgraph.Qt")
    from idscope.config import load_config
    from idscope.session import save_session, session_path

    scope = tmp_path / "demo.yaml"
    scope.write_text(DEMO_CONFIG)
    save_session(load_config(scope), scope)
    before = session_path().read_text()
    image = tmp_path / "last.png"
    assert main([str(data_dir / "a.nc"), "-e", str(image), "--size", "900x600"]) == 0
    assert image.stat().st_size > 10000
    assert session_path().read_text() == before  # an export leaves the session alone


def test_list_quantities_when_they_cannot_be_listed(data_dir, capsys, old_access_layer):
    assert main(["--list", str(data_dir / "h5"), "core_profiles"]) == 0
    out, err = capsys.readouterr()
    assert "profile   core_profiles/profiles_1d/electrons/temperature" in out
    assert "cannot list its filled quantities" in err
