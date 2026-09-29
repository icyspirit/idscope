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


@pytest.fixture
def old_access_layer(monkeypatch):
    """Data entries cannot list their filled quantities, as with Access Layer 5.4.

    Returns the real ``imas.DBEntry.list_filled_paths``.
    """
    import imas
    from imas.backends.imas_core.imas_interface import LLInterfaceError

    def list_filled_paths(*args, **kwargs):
        raise LLInterfaceError(
            "This function requires at least Access Layer version 5.7, "
            "but the current version is 5.4.3"
        )

    real = imas.DBEntry.list_filled_paths
    monkeypatch.setattr(imas.DBEntry, "list_filled_paths", list_filled_paths)
    return real


@pytest.fixture
def make_window(qtbot):
    """Create main windows, which stay referenced until the end of the test.

    pytest-qt only keeps weak references to widgets, and processes events after the
    test function returns. A window referenced by nothing else could then be
    collected while it is being painted, which with PyQt5 destroys it mid-paint
    and crashes.
    """
    pytest.importorskip("pyqtgraph.Qt")
    from idscope.gui import create_application, create_window

    create_application()
    windows = []

    def make(*args, **kwargs):
        window = create_window(*args, **kwargs)
        qtbot.addWidget(window)
        windows.append(window)
        return window

    yield make
