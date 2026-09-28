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


def test_equilibrium_template_on_dd3_data(template_window, dd3_file):
    window = template_window("equilibrium", [dd3_file])
    errors = errors_by_panel(window)
    assert set(errors) == {"Internal inductance, beta"}
    assert any("beta_normal' is not filled" in e for e in errors["Internal inductance, beta"])
    j = next(p for p in window.panels if p.config.title == "Toroidal current density")
    assert j.curves[0].data.path.text == "equilibrium/time_slice/profiles_1d/j_tor"
    flux = next(p for p in window.panels if p.config.title == "Poloidal flux")
    assert flux.kind == "map"
    assert [type(c.data).__name__ for c in flux.curves] == ["MapSeries", "ProfileSeries"]
