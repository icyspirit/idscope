import numpy as np
import pytest

from idscope.data import parse_signal_path
from idscope.errors import ConfigError
from idscope.templates import ALIASES, aliases_of, load_template, template_names


def test_template_names():
    assert template_names() == [
        "core_profiles", "core_sources", "equilibrium", "free_boundary", "nbi", "pf_active"
    ]


def test_template_aliases():
    """Templates also go by the names of the codes whose results they show."""
    assert set(ALIASES.values()) <= set(template_names())
    assert aliases_of("nbi") == ["nubdec"]
    assert aliases_of("free_boundary") == ["freegs", "sonamu"]
    assert aliases_of("equilibrium") == []
    assert load_template("nubdec").title == "nbi"
    assert load_template("SONAMU").title == "free_boundary"  # in any case
    assert load_template("FreeGS").title == "free_boundary"
    assert load_template("Equilibrium").title == "equilibrium"


@pytest.mark.parametrize("name", template_names())
def test_templates_load(name):
    config = load_template(name)
    assert config.title == name
    assert config.sources == []
    assert all(panel.signals for panel in config.panels())


def test_unknown_template():
    with pytest.raises(ConfigError, match="available: core_profiles, core_sources, "):
        load_template("nope")


@pytest.fixture
def template_window(make_window):
    def make(name, sources):
        from idscope.data.sources import parse_source

        config = load_template(name)
        config.sources = [parse_source(str(s)) for s in sources]
        win = make_window(config)
        win.resize(1300, 900)
        win.show()
        return win

    return make


def errors_by_panel(window):
    return {panel.config.title: panel.errors for panel in window.panels if panel.errors}


@pytest.mark.parametrize("name", template_names())
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
    # at one time, the currents over time are bars of their values, as at the
    # selected time
    assert currents.kind == bars.kind == "profile"
    assert currents._categories == bars._categories == COILS
    for panel in (currents, bars):
        np.testing.assert_allclose(panel._bars[0].item.opts["height"],
                                   [12, 15, 18, 18, 15, 12, 0, 0, 0, 0, 0, 0])
    assert len(section._shape_labels) == len(COILS)


SOURCES = ["nbi", "ec", "ohmic", "collisional_equipartition"]


def test_core_sources_template_on_demo_data(template_window, data_dir):
    window = template_window("core_sources", [data_dir / "a.nc", data_dir / "b.nc"])
    window.set_time(6.0)
    panels = {panel.config.title: panel for panel in window.panels}
    names = {title: [label.text for _, label in panel.legend.items]
             for title, panel in panels.items()}
    assert names["Electron heating"] == SOURCES + ["a", "b"]
    assert names["Ion heating"] == ["nbi", "collisional_equipartition", "a", "b"]
    assert names["Torque density"] == ["nbi", "a", "b"]
    assert panels["Electron heating"].plot.getAxis("bottom").labelText == "rho_tor_norm"

    # a source has the same color in every panel
    def colors(panel):
        return {c.name.split(" [")[0]: c.item.opts["pen"].color().name()
                for c in panel.curves if c.source_index == 0}

    electrons, ions = colors(panels["Electron heating"]), colors(panels["Ion heating"])
    assert ions == {name: electrons[name] for name in ions}
    assert len(set(electrons.values())) == len(SOURCES)

    bars = panels["Power at the selected time"]
    assert bars.kind == "profile" and bars._categories == SOURCES
    source = window.pool.get(window.config.sources[0])
    power = source.signal(parse_signal_path("core_sources/source[:]/global_quantities/power"))
    values, _ = power.values_at(6.0)
    # the beam, EC just on, the fading ohmic heating, and the exchange, which gives none
    np.testing.assert_allclose(1e-6 * values, [5.0, 2.98, 0.59, 0.0], atol=0.01)
    np.testing.assert_allclose(bars._bars[0].item.opts["height"], 1e-6 * values)
    np.testing.assert_allclose(bars._bars[1].item.opts["height"], 0.8e-6 * values)


NEUTRON_R, NEUTRON_Z = np.array([1.5, 2.0, 2.5]), np.array([-0.5, 0.0, 0.5, 1.0])
NEUTRON_RATES = 10.0 * NEUTRON_R[:, None] + NEUTRON_Z[None, :]  # by R and Z


@pytest.fixture(scope="module")
def beam_deposition(tmp_path_factory):
    """What a neutral beam code writes (here as NuBDeC in tbd-fusion does), at one
    time, in Data Dictionary 3.39.0: what the beam gives the plasma (one source,
    with its profiles and no totals, against rho_tor_norm and rho_pol_norm), the
    fast ions, where the ions of the beam and the neutrons are born, and the
    neutron rates. No equilibrium and no wall: they are in the entry the code was
    run on."""
    import imas
    from imas.ids_identifiers import identifiers

    from idscope.demo import structured_grid

    path = tmp_path_factory.mktemp("nbi") / "entry.nc"
    rho = np.linspace(0.0, 1.0, 41)
    with imas.DBEntry(str(path), "w", dd_version="3.39.0") as entry:
        ids = entry.factory.core_sources()
        ids.ids_properties.homogeneous_time = 1
        ids.time = [2.0]
        ids.source.resize(1)
        source = ids.source[0]
        source.identifier.name = "nbi"
        source.identifier.index = 2
        source.profiles_1d.resize(1)
        profiles = source.profiles_1d[0]
        profiles.time = 2.0
        profiles.grid.rho_tor_norm = rho
        profiles.grid.rho_pol_norm = np.sqrt(rho)
        shape = np.exp(-((rho / 0.4) ** 2))
        profiles.electrons.energy = 0.3e6 * shape
        profiles.total_ion_energy = 0.5e6 * shape
        profiles.electrons.particles = 2.0e19 * shape
        profiles.momentum_tor = 1.5 * shape
        profiles.j_parallel = 0.2e6 * shape
        entry.put(ids)

        ids = entry.factory.core_profiles()  # the fast ions, and only them
        ids.ids_properties.homogeneous_time = 1
        ids.time = [2.0]
        ids.profiles_1d.resize(1)
        profiles = ids.profiles_1d[0]
        profiles.time = 2.0
        profiles.grid.rho_tor_norm = rho
        profiles.grid.rho_pol_norm = np.sqrt(rho)
        profiles.ion.resize(1)
        profiles.ion[0].label = "D"
        profiles.ion[0].density_fast = 1.0e18 * shape
        profiles.ion[0].pressure_fast_parallel = 2.0e3 * shape
        profiles.ion[0].pressure_fast_perpendicular = 4.0e3 * shape
        entry.put(ids)

        ids = entry.factory.distribution_sources()
        ids.ids_properties.homogeneous_time = 1
        ids.time = [2.0]
        ids.source.resize(2)
        beam, neutrons = ids.source
        beam.process.resize(1)
        beam.process[0].type.name = "NBI"
        beam.markers.resize(1)
        markers = beam.markers[0]
        markers.time = 2.0
        columns = ("z", "r", "phi")  # R is found by name, wherever it is
        markers.coordinate_identifier.resize(len(columns))
        for identifier, name in zip(markers.coordinate_identifier, columns):
            member = identifiers.coordinate_identifier[name]
            identifier.name, identifier.index = member.name, member.index
        markers.positions = np.array([[0.1, 2.0, 0.0], [0.0, 2.1, 1.0], [-0.1, 2.2, 2.0]])
        markers.weights = np.full(3, 1.0e19)
        neutrons.process.resize(1)
        neutrons.process[0].type.name = "D_D_to_3He_n"
        neutrons.process[0].reactant_energy.name = "beam_thermal"
        neutrons.ggd.resize(1)
        ggd = neutrons.ggd[0]
        ggd.time = 2.0
        structured_grid(ggd.grid, (("r", NEUTRON_R), ("z", NEUTRON_Z)))
        ggd.particles.resize(1)
        ggd.particles[0].grid_index = 1
        ggd.particles[0].grid_subset_index = 1  # the nodes
        ggd.particles[0].values = NEUTRON_RATES.ravel(order="F")  # R running fastest
        entry.put(ids)

        ids = entry.factory.summary()
        ids.ids_properties.homogeneous_time = 1
        ids.time = [2.0]
        rates = ids.fusion.neutron_rates
        rates.dd.beam_thermal.value = [0.9e15]
        rates.dd.thermal.value = [0.1e15]
        rates.dd.total.value = [1.0e15]
        rates.dt.total.value = [1.0e13]
        rates.total.value = [1.01e15]
        entry.put(ids)
    return path


def test_core_sources_template_on_a_beam_deposition(template_window, beam_deposition):
    window = template_window("core_sources", [beam_deposition])
    # the profiles, at the one time; the totals are not filled
    assert set(errors_by_panel(window)) == {
        "Power", "Driven current", "Torque", "Power at the selected time"
    }
    profiles = window.panels[:5]
    assert all([c.name for c in panel.curves] == ["nbi"] for panel in profiles)
    torque = profiles[4].curves[0].data
    assert torque.path.text == "core_sources/source[:]/profiles_1d/momentum_tor"
    assert window.time == 2.0
    x, y = profiles[0].curves[0].item.getData()
    np.testing.assert_allclose(x, np.linspace(0.0, 1.0, 41))
    assert y[0] == pytest.approx(0.3)


def n_points(item) -> int:
    x, _ = item.getData()
    return 0 if x is None else len(x)


def test_nbi_template_on_demo_data(template_window, data_dir):
    window = template_window("nbi", [data_dir / "a.nc", data_dir / "b.nc"])
    window.set_time(6.0)
    panels = {panel.config.title: panel for panel in window.panels}
    names = {title: [label.text for _, label in panel.legend.items]
             for title, panel in panels.items()}
    assert names["Heating"] == [f"{name} electrons" for name in SOURCES] + [
        "nbi ions", "collisional_equipartition ions", "a", "b"
    ]
    assert names["Fast ion density"] == ["D", "a", "b"]
    assert names["Fast ion pressures"] == ["D parallel", "D perpendicular", "a", "b"]
    assert names["Neutron rates"] == [
        "total", "D-D beam-thermal", "D-D thermal", "D-T", "a", "b"
    ]

    # where the ions of the beam are born, as dots, from 3 s on
    births = [c for c in panels["Births of the beam ions"].curves if c.label == "births"]
    assert births[0].item.opts["pen"] is None and births[0].item.opts["symbol"] == "o"
    assert [n_points(c.item) for c in births] == [400, 400]
    window.set_time(2.0)
    assert [n_points(c.item) for c in births] == [0, 0]

    # where the neutrons are born, a reaction at a time
    window.set_time(6.0)
    neutrons = panels["Births of the neutrons"]
    selector = neutrons.image_selector
    assert [selector.itemText(i) for i in range(selector.count())] == [
        f"D_D_to_3He_n {energy} births [{source}]"
        for source in ("a", "b") for energy in ("thermal_thermal", "beam_thermal")
    ]
    assert neutrons._image.image.shape == (25, 49)  # R by Z
    selector.setCurrentIndex(1)
    assert neutrons._image.image.shape == (25, 49)


def test_nbi_template_on_a_beam_code_output(template_window, beam_deposition):
    window = template_window("nbi", [beam_deposition])
    panels = {panel.config.title: panel for panel in window.panels}
    # no equilibrium and no wall: the births are drawn all the same
    errors = errors_by_panel(window)
    assert set(errors) == {"Births of the beam ions", "Births of the neutrons"}
    assert len(errors["Births of the beam ions"]) == 3  # psi, boundary, wall
    assert window.time == 2.0
    births = next(c for c in panels["Births of the beam ions"].curves if c.label == "births")
    x, y = births.item.getData()
    np.testing.assert_allclose(x, [2.0, 2.1, 2.2])
    np.testing.assert_allclose(y, [0.1, 0.0, -0.1])
    neutrons = panels["Births of the neutrons"]
    assert neutrons.image_selector.itemText(0) == "D_D_to_3He_n beam_thermal births"
    np.testing.assert_allclose(neutrons._image.image, NEUTRON_RATES)
    fast = panels["Fast ion density"].curves[0]
    assert fast.name == "D"
    assert fast.item.getData()[1][0] == pytest.approx(1.0)
    # the rates at the one time, as bars of their values
    rates = panels["Neutron rates"]
    assert rates.kind == "profile"
    assert rates._categories == ["total", "D-D beam-thermal", "D-D thermal", "D-T"]
    np.testing.assert_allclose([bar.item.opts["height"][0] for bar in rates._bars],
                               [1.01e15, 0.9e15, 0.1e15, 1.0e13])
    assert [label.textItem.toPlainText() for label in rates._value_labels] == [
        "1.01e+15", "9e+14", "1e+14", "1e+13"
    ]


@pytest.fixture(scope="module")
def free_boundary_solution(tmp_path_factory):
    """What a free-boundary equilibrium solver writes (here as SONAMU and FreeGS
    in tbd-fusion do), in Data Dictionary 3.39.0: the equilibrium at one time,
    without rho_tor_norm, and the coil currents that hold it."""
    import imas

    from idscope.demo import equilibrium, pf_active

    path = tmp_path_factory.mktemp("free_boundary") / "entry.nc"
    with imas.DBEntry(str(path), "w", dd_version="3.39.0") as entry:
        entry.put(equilibrium(entry.factory, time=5.0, rho_tor_norm=False))
        entry.put(pf_active(entry.factory, time=5.0))
    return path


def test_free_boundary_template_on_a_solution(template_window, free_boundary_solution):
    window = template_window("free_boundary", [free_boundary_solution])
    section, bars, *profiles = window.panels
    # the solver writes no wall
    assert set(errors_by_panel(window)) == {"Poloidal cross-section"}
    assert "'wall'" in section.errors[0]
    labels = [label.text for _, label in section.legend.items]
    assert labels == ["boundary", "X-points", "X-points to hold", "magnetic axis", "coils"]
    shapes = {c.label: c for c in section.curves}
    x_point = shapes["X-points"].item.getData()
    target = shapes["X-points to hold"].item.getData()
    np.testing.assert_allclose(np.ravel(x_point), [1.8 + 0.03 * np.sin(5.0) - 0.15, -0.8])
    np.testing.assert_allclose(np.ravel(target), np.ravel(x_point) + [-0.02, -0.04])
    assert len(section._shape_labels) == len(COILS)
    assert bars.kind == "profile" and bars._categories == COILS
    np.testing.assert_allclose(bars._bars[0].item.opts["height"], FLAT_TOP)
    # against psi normalized from the magnetic axis to the boundary
    assert {p.plot.getAxis("bottom").labelText for p in profiles} == {"psi_n"}
    x, _, _ = profiles[0].curves[0].data.at(5.0)
    np.testing.assert_allclose(x, np.linspace(0.0, 1.0, 51) ** 2)


def test_free_boundary_template_on_demo_data(template_window, data_dir):
    """Data Dictionary 4: the X-point is a critical point of the contour tree."""
    window = template_window("free_boundary", [data_dir / "a.nc"])
    window.set_time(5.0)
    section = window.panels[0]
    x_points = next(c for c in section.curves if c.label == "X-points")
    assert x_points.data.path.text == "equilibrium/time_slice/contour_tree/node[1:]"
    np.testing.assert_allclose(np.ravel(x_points.item.getData()),
                               [1.8 + 0.03 * np.sin(5.0) - 0.15, -0.8])
