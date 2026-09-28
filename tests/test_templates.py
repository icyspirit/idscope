import numpy as np
import pytest

from idscope.errors import ConfigError
from idscope.templates import load_template, template_names


def test_template_names():
    assert template_names() == ["core_profiles", "equilibrium"]


@pytest.mark.parametrize("name", ["core_profiles", "equilibrium"])
def test_templates_load(name):
    config = load_template(name)
    assert config.title == name
    assert config.sources == []
    assert all(panel.signals for panel in config.panels())


def test_unknown_template():
    with pytest.raises(ConfigError, match="available: core_profiles, equilibrium"):
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


@pytest.mark.parametrize("name", ["core_profiles", "equilibrium"])
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
