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
