import numpy as np
import pytest

from idscope.errors import ConfigError
from idscope.templates import load_template, template_names


def test_template_names():
    assert template_names() == ["core_profiles", "equilibrium", "pf_active"]


@pytest.mark.parametrize("name", ["core_profiles", "equilibrium", "pf_active"])
def test_templates_load(name):
    config = load_template(name)
    assert config.title == name
    assert config.sources == []
    assert all(panel.signals for panel in config.panels())


def test_unknown_template():
    with pytest.raises(ConfigError, match="available: core_profiles, equilibrium, pf_active"):
        load_template("nope")


@pytest.fixture
def template_window(qtbot):
    pytest.importorskip("pyqtgraph.Qt")
    from idscope.gui import create_application, create_window

    create_application()

    def make(name, sources):
        from idscope.data.sources import parse_source

        config = load_template(name)
        config.sources = [parse_source(str(s)) for s in sources]
        win = create_window(config)
        qtbot.addWidget(win)
        win.resize(1300, 900)
        win.show()
        return win

    return make


def errors_by_panel(window):
    return {panel.config.title: panel.errors for panel in window.panels if panel.errors}


@pytest.mark.parametrize("name", ["core_profiles", "equilibrium", "pf_active"])
def test_templates_on_demo_data(template_window, data_dir, name):
    window = template_window(name, [data_dir / "a.nc", data_dir / "b.nc"])
    assert errors_by_panel(window) == {}
    assert all(panel.curves for panel in window.panels)


def test_core_profiles_template_on_dd3_data(template_window, dd3_file):
    window = template_window("core_profiles", [dd3_file])
    # The training data has no core_profiles/global_quantities
    assert set(errors_by_panel(window)) == {
        "Loop voltage", "Internal inductance, poloidal beta"
    }
    ip = window.panels[0].curves[0].data
    assert ip.path.ids_name == "equilibrium"  # the alternative was used
    ti = window.panels[1].curves[1].data
    assert ti.path.text == "core_profiles/profiles_1d/t_i_average[0]"


def x_titles(window):
    return {panel.plot.getAxis("bottom").labelText for panel in window.panels}


def test_equilibrium_template_on_dd3_data(template_window, dd3_file):
    window = template_window("equilibrium", [dd3_file])
    assert all(panel.kind == "profile" for panel in window.panels)
    # The training data has no lower triangularity profile
    errors = errors_by_panel(window)
    assert set(errors) == {"Shape"}
    assert "triangularity_lower' is not filled" in errors["Shape"][0]
    j = next(p for p in window.panels if p.config.title == "Current density")
    assert j.curves[0].data.path.text == "equilibrium/time_slice/profiles_1d/j_tor"
    assert x_titles(window) == {"rho_tor_norm"}


@pytest.fixture(scope="module")
def no_rho_file(tmp_path_factory):
    """Demo data without rho_tor_norm (in core_profiles and equilibrium)."""
    from idscope.demo import write_entry

    path = tmp_path_factory.mktemp("no_rho") / "no_rho.nc"
    write_entry(str(path), 0.8, rho_tor_norm=False)
    return path


def test_equilibrium_template_without_rho_tor_norm(template_window, no_rho_file):
    window = template_window("equilibrium", [no_rho_file])
    assert errors_by_panel(window) == {}
    # profiles are drawn against psi normalized from the axis to the boundary
    assert x_titles(window) == {"psi_n"}
    x, _, _ = window.panels[0].curves[0].data.at(5.0)
    np.testing.assert_allclose(x, np.linspace(0.0, 1.0, 51) ** 2)
    # one shared x axis per column
    for panels in window.column_panels:
        master = panels[0].plot.getViewBox()
        assert all(p.plot.getViewBox().linkedView(0) is master for p in panels[1:])


def test_core_profiles_template_without_rho_tor_norm(template_window, no_rho_file):
    window = template_window("core_profiles", [no_rho_file])
    assert errors_by_panel(window) == {}
    traces, profiles, images = window.column_panels
    assert {p.plot.getAxis("bottom").labelText for p in profiles} == {"psi_n"}
    assert {p.plot.getAxis("left").labelText for p in images} == {"psi_n"}
    x, _, _ = profiles[0].curves[0].data.at(5.0)
    np.testing.assert_allclose(x, np.linspace(0.0, 1.0, 51) ** 2)


COILS = ["CS3U", "CS2U", "CS1U", "CS1L", "CS2L", "CS3L",
         "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]
FLAT_TOP = np.array([-6, -9, -12, -12, -9, -6, -4, -6, 8, 8, -6, -4])  # kA


def test_pf_active_template_on_demo_data(template_window, data_dir):
    window = template_window("pf_active", [data_dir / "a.nc", data_dir / "b.nc"])
    window.set_time(5.0)
    currents, voltages, bars, section = window.panels
    assert len(currents.curves) == len(voltages.curves) == 24
    assert [label.text for _, label in currents.legend.items] == COILS + ["a", "b"]
    assert bars.kind == "profile" and bars._categories == COILS
    np.testing.assert_allclose(bars._bars[0].item.opts["height"], FLAT_TOP)
    np.testing.assert_allclose(bars._bars[1].item.opts["height"], 0.8 * FLAT_TOP)
    labels = [label.text for _, label in section.legend.items]
    assert labels == ["boundary", "magnetic axis", "coils", "wall", "a", "b"]
    coils = next(c for c in section.curves if c.label == "coils")
    x, _ = coils.item.getData()
    assert np.isnan(x).sum() == len(COILS)  # one outline per coil
    assert sorted(t.textItem.toPlainText() for t in section._shape_labels) == sorted(COILS)
    assert section.plot.getAxis("bottom").labelText == "R [m]"


@pytest.fixture(scope="module")
def solved_coils(tmp_path_factory):
    """Coil currents at one time and no voltages, as an equilibrium code that solves
    for the currents writes them (here in Data Dictionary 3.39.0)."""
    import imas

    from idscope.demo import pf_active

    path = tmp_path_factory.mktemp("solved") / "coils.nc"
    with imas.DBEntry(str(path), "w", dd_version="3.39.0") as entry:
        ids = pf_active(entry.factory, time=0.0)
        for coil in ids.coil:
            coil.voltage.data = []
        entry.put(ids)
    return path


def test_pf_active_template_on_one_time(template_window, solved_coils):
    window = template_window("pf_active", [solved_coils])
    currents, voltages, bars, section = window.panels
    # no voltages, no equilibrium and no wall: the coils are drawn all the same
    assert set(errors_by_panel(window)) == {"Coil voltages", "Poloidal cross-section"}
    assert all(c.item.opts["symbol"] == "o" for c in currents.curves)  # one sample
    np.testing.assert_allclose(bars._bars[0].item.opts["height"],
                               [12, 15, 18, 18, 15, 12, 0, 0, 0, 0, 0, 0])
    assert len(section._shape_labels) == len(COILS)
