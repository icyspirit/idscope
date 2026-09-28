import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def user_config_dir(tmp_path, monkeypatch):
    """Keep idscope's settings and last session away from the user's."""
    directory = tmp_path / "user_config"
    monkeypatch.setenv("IDSCOPE_CONFIG_DIR", str(directory))
    return directory


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    """Synthetic IMAS data: two netCDF files and one HDF5 entry."""
    from idscope.demo import write_entry

    directory = tmp_path_factory.mktemp("data")
    write_entry(str(directory / "a.nc"), 1.0)
    write_entry(str(directory / "b.nc"), 0.8)
    write_entry(f"imas:hdf5?path={directory / 'h5'}", 1.2)
    return directory


@pytest.fixture(scope="session")
def dd3_file(tmp_path_factory):
    """The ITER 134173/106 training data (DD 3.39.0) as a netCDF file."""
    import imas
    import imas.training

    path = tmp_path_factory.mktemp("dd3") / "iter_134173_106.nc"
    training = imas.training.get_training_db_entry()
    with imas.DBEntry(str(path), "w", dd_version="3.39.0") as entry:
        for name in ("core_profiles", "equilibrium"):
            entry.put(training.get(name, autoconvert=False))
    training.close()
    return path


@pytest.fixture
def source_a(data_dir):
    from idscope.data import DataSource, parse_source

    source = DataSource(parse_source(f"A={data_dir / 'a.nc'}"))
    yield source
    source.close()
