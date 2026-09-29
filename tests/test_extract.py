import numpy as np
import pytest

from idscope.data import (
    Channels,
    MapSeries,
    ProfileSeries,
    Shapes,
    Trace,
    parse_signal_path,
)
from idscope.data.extract import geometry_outline, normalize, plan_path
from idscope.data.sources import DataSource, parse_source
from idscope.errors import NoDataError, PathError

TIME = np.linspace(0.0, 10.0, 41)
RAMP = np.minimum(TIME / 2.0, 1.0)


def get(source, text, x=None):
    return source.signal(parse_signal_path(text), parse_signal_path(x) if x else None)


def test_trace_from_time_aos(source_a):
    data = get(source_a, "equilibrium/time_slice/global_quantities/ip")
    assert isinstance(data, Trace)
    np.testing.assert_allclose(data.time, TIME)
    np.testing.assert_allclose(data.values, -1e6 * RAMP)
    assert data.value_axis.units == "A"
    assert data.value_axis.label == "global_quantities/ip"
    assert data.x_axis.title() == "time [s]"


@pytest.mark.parametrize(
    "text", ["core_profiles/global_quantities/ip", "summary/global_quantities/ip/value"]
)
def test_trace_from_homogeneous_time_array(source_a, text):
    data = get(source_a, text)
    assert isinstance(data, Trace)
    np.testing.assert_allclose(data.time, TIME)
    np.testing.assert_allclose(data.values, 1e6 * RAMP)


def test_trace_heterogeneous_time(source_a):
    data = get(source_a, "magnetics/flux_loop[1]/flux/data")
    assert isinstance(data, Trace)
    assert data.time.shape == (2001,)
    assert data.values.shape == (2001,)
    ip = get(source_a, "magnetics/ip(1)/data")
    assert ip.time.shape == (201,)
    np.testing.assert_allclose(ip.values[-1], 1e6)


def test_trace_from_profile_index(source_a):
    data = get(source_a, "core_profiles/profiles_1d/electrons/temperature[0]")
    assert isinstance(data, Trace)
    assert data.values.shape == (41,)
    profile = get(source_a, "core_profiles/profiles_1d/electrons/temperature")
    np.testing.assert_allclose(data.values, [y[0] for y in profile.ys])


def test_profile_series(source_a):
    data = get(source_a, "core_profiles/profiles_1d/electrons/temperature")
    assert isinstance(data, ProfileSeries)
    assert len(data.xs) == len(data.ys) == 41
    assert data.x_axis.label == "rho_tor_norm"
    assert data.value_axis.units == "eV"
    x, y, t = data.at(5.1)
    assert t == pytest.approx(5.0)
    np.testing.assert_allclose(x, np.linspace(0, 1, 51))
    assert y[0] > y[-1]
    time, grid, z = data.as_image()
    assert z.shape == (41, 51)


def test_profile_with_missing_slices(source_a):
    data = get(source_a, "core_profiles/profiles_1d/ion[1]/temperature")
    assert isinstance(data, ProfileSeries)
    assert [len(y) for y in data.ys[-4:]] == [51, 0, 0, 0]
    x, y, t = data.at(10.0)
    assert len(y) == 0 and t == pytest.approx(10.0)


def test_profile_non_rho_coordinate(source_a):
    data = get(source_a, "equilibrium/time_slice/profiles_1d/q")
    assert data.x_axis.label == "psi"


def test_profile_with_x_path(source_a):
    data = get(
        source_a,
        "core_profiles/profiles_1d/electrons/temperature",
        x="core_profiles/profiles_1d/grid/psi",
    )
    np.testing.assert_allclose(data.xs[0], 20 * np.linspace(0, 1, 51) ** 2)
    assert data.x_axis.title() == "psi [Wb]"
    with pytest.raises(PathError):
        get(source_a, "core_profiles/global_quantities/ip",
            x="core_profiles/profiles_1d/grid/psi")


def test_normalized_profile(source_a):
    rho = np.linspace(0, 1, 51)
    psi_n = get(source_a, "norm(core_profiles/profiles_1d/grid/psi)")
    assert isinstance(psi_n, ProfileSeries)
    assert len(psi_n.ys) == 41
    np.testing.assert_allclose(psi_n.ys[3], rho**2)
    assert psi_n.value_axis.title() == "psi_n"
    te = get(
        source_a,
        "core_profiles/profiles_1d/electrons/temperature",
        x="norm(core_profiles/profiles_1d/grid/psi)",
    )
    np.testing.assert_allclose(te.xs[0], rho**2)
    assert te.x_axis.title() == "psi_n"


def test_normalize_errors(source_a):
    with pytest.raises(PathError, match="applies to profiles"):
        get(source_a, "norm(core_profiles/global_quantities/ip)")
    q = get(source_a, "core_profiles/profiles_1d/q")
    flat = ProfileSeries(q.path, q.time, q.xs, [np.ones_like(y) for y in q.ys],
                         q.x_axis, q.value_axis)
    with pytest.raises(NoDataError, match="first and last values are missing or equal"):
        normalize(flat)
    # Slices that cannot be normalized are left empty (NaN), the others are kept
    flat.ys[0] = q.ys[0]
    ys = normalize(flat).ys
    assert np.isfinite(ys[0]).all() and np.isnan(ys[1]).all()


def test_map_series(source_a):
    data = get(source_a, "equilibrium/time_slice/profiles_2d[0]/psi")
    assert isinstance(data, MapSeries)
    assert data.x_axis.label == "grid/dim1"
    assert data.x2_axis.label == "grid/dim2"
    x1, x2, z, t = data.at(3.0)
    assert x1.shape == (45,) and x2.shape == (81,) and z.shape == (45, 81)
    assert t == pytest.approx(3.0)


def test_transpose(source_a):
    psi = get(source_a, "equilibrium/time_slice/profiles_2d[0]/psi")
    x1, x2, z, _ = psi.at(3.0)
    for text in ("transpose(equilibrium/time_slice/profiles_2d[0]/psi)",
                 "transpose(equilibrium/time_slice/profiles_2d[0]/psi, 1, 0)"):
        swapped = get(source_a, text)
        assert isinstance(swapped, MapSeries)
        assert (swapped.x_axis.label, swapped.x2_axis.label) == ("grid/dim2", "grid/dim1")
        y1, y2, w, t = swapped.at(3.0)
        np.testing.assert_allclose(y1, x2)
        np.testing.assert_allclose(y2, x1)
        np.testing.assert_allclose(w, z.T)
        assert t == pytest.approx(3.0)
    same = get(source_a, "transpose(equilibrium/time_slice/profiles_2d[0]/psi, 0, 1)")
    np.testing.assert_allclose(same.at(3.0)[2], z)
    with pytest.raises(PathError, match="has 1 dimension left, nothing to transpose"):
        get(source_a, "transpose(core_profiles/profiles_1d/electrons/temperature)")
    with pytest.raises(PathError, match="new order of the 2 dimensions \\(grid/dim1, "
                                        "grid/dim2\\).*not 0, 0"):
        get(source_a, "transpose(equilibrium/time_slice/profiles_2d[0]/psi, 0, 0)")


@pytest.fixture(scope="module")
def bolometer_source(tmp_path_factory):
    """A radiated power density on a (dim1, dim2) grid, with time as its third
    dimension."""
    import imas
    from imas.ids_defs import IDS_TIME_MODE_HETEROGENEOUS

    path = tmp_path_factory.mktemp("bolometer") / "bolometer.nc"
    with imas.DBEntry(str(path), "w") as entry:
        bolo = entry.factory.bolometer()
        bolo.ids_properties.homogeneous_time = IDS_TIME_MODE_HETEROGENEOUS
        bolo.grid.dim1 = np.linspace(1.0, 2.0, 5)  # the coordinates of the density
        bolo.grid.dim2 = np.linspace(-1.0, 1.0, 7)
        density = bolo.power_density
        density.time = np.array([0.0, 1.0, 2.0, 3.0])
        i, j, k = np.meshgrid(range(5), range(7), range(4), indexing="ij")
        density.data = 100.0 * i + 10.0 * j + k  # (dim1, dim2, time)
        entry.put(bolo)
    source = DataSource(parse_source(str(path)))
    yield source
    source.close()


def test_dimensions_of_data_with_time(bolometer_source):
    path = "bolometer/power_density/data"  # (dim1, dim2, time)
    # Time is a dimension like the others: three are too many to show
    with pytest.raises(PathError, match="3 dimensions \\(grid/dim1, grid/dim2, time\\)"):
        get(bolometer_source, path)
    # An index selects along the dimensions in the order of the Data Dictionary
    frozen = get(bolometer_source, f"{path}[:, :, 2]")  # the map at the third time
    assert isinstance(frozen, MapSeries) and frozen.time is None
    assert frozen.fixed_time == 2.0
    x1, x2, z, _ = frozen.at(None)
    assert (x1.shape, x2.shape, z[3, 4]) == ((5,), (7,), 342.0)
    column = get(bolometer_source, f"{path}[:, 4]")  # a map of dim1 (x) and time (y)
    assert isinstance(column, MapSeries) and column.time is None
    assert (column.x_axis.label, column.x2_axis.label) == ("grid/dim1", "time")
    x1, x2, z, _ = column.at(None)
    np.testing.assert_allclose(x2, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(z[:, 2], [42, 142, 242, 342, 442])
    trace = get(bolometer_source, f"{path}[3, 4]")  # a curve along time
    assert isinstance(trace, Trace)
    np.testing.assert_allclose(trace.values, [340, 341, 342, 343])
    # transpose() reorders the dimensions, time as any other
    image = get(bolometer_source, f"transpose({path}[:, 4])")
    assert (image.x_axis.label, image.x2_axis.label) == ("time", "grid/dim1")
    assert image.at(None)[2].shape == (4, 5)
    for axes in ("", ", 1, 0", ", (1, 0)", ", [1, 0]"):
        swapped = get(bolometer_source, f"transpose({path}[:, :, 2]{axes})")
        assert (swapped.x_axis.label, swapped.x2_axis.label) == ("grid/dim2", "grid/dim1")
        assert swapped.at(None)[2][4, 3] == 342.0
    with pytest.raises(PathError, match="3 dimensions \\(time, grid/dim1, grid/dim2\\)"):
        get(bolometer_source, f"transpose({path}, 2, 0, 1)")
    # Without axes, the order is reversed, as numpy.transpose: (2, 1, 0) for three
    ids = bolometer_source.get_ids("bolometer")
    plan = plan_path(ids.metadata, parse_signal_path(f"transpose({path})"))
    assert plan.axes == (2, 1, 0)


def test_fixed_time_slice(source_a):
    data = get(source_a, "core_profiles/profiles_1d[4]/electrons/temperature")
    assert isinstance(data, ProfileSeries)
    assert data.time is None
    assert data.fixed_time == pytest.approx(TIME[4])
    x, y, t = data.at(9.0)
    assert len(y) == 51 and t == pytest.approx(TIME[4])


@pytest.mark.parametrize(
    "text, message",
    [
        ("magnetics/flux_loop/flux/data", "does not depend on time"),
        ("core_profiles/profiles_1d/electrons/temprature", "did you mean 'temperature'"),
        ("core_profiles/profiles_1d/electrons", "is a structure"),
        ("magnetics/flux_loop[0]/name", "only numbers"),
        ("core_profiles/profiles_1d/electrons[0]/temperature", "cannot be indexed"),
        ("equilibrium/time_slice/profiles_2d", "array of structures"),
    ],
)
def test_path_errors(source_a, text, message):
    with pytest.raises(PathError, match=message):
        get(source_a, text)


@pytest.mark.parametrize(
    "text, message",
    [
        ("magnetics/flux_loop[5]/flux/data", "out of range"),
        ("core_profiles/profiles_1d/electrons/pressure", "not filled"),
        ("interferometer/channel[0]/n_e_line/data", "interferometer"),
    ],
)
def test_no_data_errors(source_a, text, message):
    with pytest.raises(NoDataError, match=message):
        get(source_a, text)


def test_hdf5_source(data_dir):
    source = DataSource(parse_source(f"imas:hdf5?path={data_dir / 'h5'}"))
    try:
        assert source.label == "h5"
        data = get(source, "equilibrium/time_slice/global_quantities/ip")
        np.testing.assert_allclose(data.values, -1.2e6 * RAMP)
        flux = get(source, "magnetics/flux_loop[1]/flux/data")
        assert flux.time.shape == (2001,)
    finally:
        source.close()


def test_other_major_dd_version(dd3_file):
    source = DataSource(parse_source(str(dd3_file)))
    try:
        te = get(source, "core_profiles/profiles_1d/electrons/temperature")
        assert len(te.ys) == 3 and len(te.ys[0]) == 101
        assert any("3.39.0" in note for note in source.notes)
        psi = get(source, "equilibrium/time_slice/profiles_2d[0]/psi")
        assert psi.at(0.0)[2].shape == (65, 129)
        # psi decreases from the magnetic axis to the boundary in this data
        psi_n = get(source, "norm(equilibrium/time_slice/profiles_1d/psi)")
        for y in psi_n.ys:
            assert y[0] == 0.0 and y[-1] == 1.0 and np.all(np.diff(y) > 0)
    finally:
        source.close()


PF_FLAT_TOP = [-6e3, -9e3, -12e3, -12e3, -9e3, -6e3, -4e3, -6e3, 8e3, 8e3, -6e3, -4e3]
COILS = ["CS3U", "CS2U", "CS1U", "CS1L", "CS2L", "CS3L",
         "PF1", "PF2", "PF3", "PF4", "PF5", "PF6"]


def test_wildcard_traces(source_a):
    currents = get(source_a, "pf_active/coil[:]/current/data")
    assert isinstance(currents, Channels) and currents.kind == "trace"
    assert currents.names == COILS
    assert currents.axis_label == "coil"
    assert all(member.time.shape == (41,) for member in currents.members)
    values, t = currents.values_at(5.1)
    assert t == pytest.approx(5.0)
    np.testing.assert_allclose(values, PF_FLAT_TOP)
    assert get(source_a, "pf_active/coil[6:]/current/data").names == COILS[6:]
    assert get(source_a, "pf_active/coil(7:12)/current/data").names == COILS[6:]


def test_index_on_the_data_of_every_element(source_a):
    # The index of the data node selects along the data (time), for every coil
    first = get(source_a, "pf_active/coil[:]/current/data[0]")
    assert isinstance(first, Channels) and first.names == COILS
    assert all(member.time is None for member in first.members)
    assert first.members[0].fixed_time == 0.0
    np.testing.assert_allclose(
        [float(m.values) for m in first.members[:3]], [12e3, 15e3, 18e3]
    )
    third = get(source_a, "pf_active/coil[:]/current/data[2]")  # t = 0.5 s
    assert third.members[0].fixed_time == pytest.approx(0.5)
    np.testing.assert_allclose(
        [float(m.values) for m in third.members[:3]], [7.5e3, 9e3, 10.5e3]
    )
    last_coil = get(source_a, "pf_active/coil[-1]/current/data")
    assert isinstance(last_coil, Trace) and last_coil.values[-1] == pytest.approx(-4e3)
    # current/data has one dimension: the wildcard does not add one
    with pytest.raises(PathError, match="'coil/current/data' has 1 dimension, but 2"):
        get(source_a, "pf_active/coil[:]/current/data[:, 0]")
    with pytest.raises(PathError, match="or 'coil\\[:\\]' for all of them"):
        get(source_a, "pf_active/coil/current/data")
    with pytest.raises(NoDataError, match="Index 20 is out of range for 'coil'"):
        get(source_a, "pf_active/coil[20]/current/data")
    with pytest.raises(NoDataError, match="Index 99 is out of range for 'coil/current"):
        get(source_a, "pf_active/coil[:]/current/data[99]")


def test_wildcards_below_the_time_axis(source_a):
    ions = get(source_a, "core_profiles/profiles_1d/ion[:]/temperature")
    assert isinstance(ions, Channels) and ions.kind == "profile"
    assert ions.names == ["D", "C"]
    # the second species is missing in the last three time slices
    assert [len(y) for y in ions.members[1].ys[-4:]] == [51, 0, 0, 0]
    central = get(source_a, "core_profiles/profiles_1d/ion[:]/temperature[0]")
    assert isinstance(central, Channels) and central.names == ["D", "C"]
    assert central.kind == "trace" and central.members[0].values.shape == (41,)
    np.testing.assert_allclose(central.members[0].values,
                               [y[0] if len(y) else np.nan for y in ions.members[0].ys])
    with pytest.raises(PathError, match="'profiles_1d/ion/temperature' has 1 dimension"):
        get(source_a, "core_profiles/profiles_1d/ion[:]/temperature[:, 0]")
    psi_n = get(source_a, "core_profiles/profiles_1d/ion[:]/temperature",
                x="norm(core_profiles/profiles_1d/grid/psi)")
    assert isinstance(psi_n, Channels) and psi_n.x_axis.title() == "psi_n"


def test_time_axis_ranges_and_cuts(source_a):
    # The time array of structures is the time axis, not a wildcard, even with [:]
    implicit = get(source_a, "core_profiles/profiles_1d/electrons/temperature[0]")
    for text in ("core_profiles/profiles_1d[:]/electrons/temperature[0]",
                 "core_profiles/profiles_1d(:)/electrons/temperature(1)",
                 "core_profiles/profiles_1d[:]/electrons/temperature[0]"):
        te0 = get(source_a, text)
        assert isinstance(te0, Trace)
        np.testing.assert_allclose(te0.values, implicit.values)
    with pytest.raises(PathError, match="has 1 dimension, but 2 indices are given"):
        get(source_a, "core_profiles/profiles_1d[:]/electrons/temperature[:, 0]")
    last = get(source_a, "core_profiles/profiles_1d[-1]/electrons/temperature")
    assert last.time is None and last.fixed_time == pytest.approx(10.0)
    some = get(source_a, "core_profiles/profiles_1d[0:10]/electrons/temperature[0]")
    np.testing.assert_allclose(some.time, TIME[:10])
    np.testing.assert_allclose(some.values, implicit.values[:10])
    cut = get(source_a, "equilibrium/time_slice/profiles_2d[0]/psi[:, 40]")  # Z = 0
    assert isinstance(cut, ProfileSeries) and cut.x_axis.label == "grid/dim1"
    r, psi, _ = cut.at(5.0)
    assert r.shape == psi.shape == (45,)
    assert np.argmin(psi) == np.argmin(np.abs(r - (1.8 + 0.03 * np.sin(5.0))))


def test_shapes(source_a):
    coils = get(source_a, "pf_active/coil[:]/element[:]/geometry")
    assert isinstance(coils, Shapes) and coils.shape == "geometry"
    assert coils.time is None and coils.names == COILS
    outlines, _ = coils.at(3.0)
    np.testing.assert_allclose(  # PF1, a rectangle, closed
        outlines[6], [[1.1, 1.45], [1.4, 1.45], [1.4, 1.75], [1.1, 1.75], [1.1, 1.45]]
    )
    np.testing.assert_allclose(outlines[0][[0, -1]], [[0.65, 1.0], [0.65, 1.0]])  # outline
    wall = get(source_a, "wall/description_2d[0]/limiter/unit[:]/outline")
    assert wall.shape == "outline" and wall.names == ["first wall"]
    points = wall.at(None)[0][0]
    assert len(points) == 121  # closed: its ends are one segment apart
    boundary = get(source_a, "equilibrium/time_slice/boundary/outline")
    assert boundary.is_time_dependent and len(boundary.slices) == 41
    axis = get(source_a, "equilibrium/time_slice/global_quantities/magnetic_axis")
    (point,), t = axis.at(5.0)
    assert axis.shape == "point" and t == pytest.approx(5.0)
    np.testing.assert_allclose(point, [[1.82 + 0.03 * np.sin(5.0), 0.01 * np.sin(10.0)]])
    with pytest.raises(PathError, match="is a structure; continue the path"):
        get(source_a, "equilibrium/time_slice/boundary")
    parts = get(source_a, "wall/description_2d[0]/limiter/unit[:]/outline/z",
                x="wall/description_2d[0]/limiter/unit[:]/outline/r")
    assert isinstance(parts, Channels) and parts.x_axis.label == "r"


def test_geometry_types():
    import imas

    coil = imas.IDSFactory().pf_active().coil
    coil.resize(1)
    coil[0].element.resize(1)
    geometry = coil[0].element[0].geometry
    assert geometry_outline(geometry) is None  # nothing filled
    geometry.rectangle.r, geometry.rectangle.z = 2.0, 0.0
    geometry.rectangle.width, geometry.rectangle.height = 0.2, 0.4
    # without geometry_type, whatever is filled
    np.testing.assert_allclose(geometry_outline(geometry)[:4],
                               [[1.9, -0.2], [2.1, -0.2], [2.1, 0.2], [1.9, 0.2]])
    geometry.geometry_type = 3
    oblique = geometry.oblique
    oblique.r, oblique.z, oblique.length_alpha, oblique.length_beta = 1.0, 0.0, 0.2, 0.1
    oblique.alpha, oblique.beta = 0.0, np.pi / 4
    shift = 0.1 * np.sqrt(0.5)
    np.testing.assert_allclose(geometry_outline(geometry)[:4], [
        [1.0, 0.0], [1.2, 0.0], [1.2 - shift, shift], [1.0 - shift, shift],
    ], atol=1e-12)
    geometry.geometry_type = 5
    geometry.annulus.r, geometry.annulus.z = 2.0, 0.0
    geometry.annulus.radius_inner, geometry.annulus.radius_outer = 0.1, 0.2
    ring = geometry_outline(geometry)
    gap = np.flatnonzero(np.isnan(ring[:, 0]))
    assert len(gap) == 1
    radii = np.hypot(ring[:, 0] - 2.0, ring[:, 1])
    np.testing.assert_allclose(radii[: gap[0]], 0.2)
    np.testing.assert_allclose(radii[gap[0] + 1:], 0.1)
    geometry.geometry_type = 6
    line = geometry.thick_line
    line.first_point.r, line.first_point.z = 1.0, 0.0
    line.second_point.r, line.second_point.z = 1.0, 1.0
    line.thickness = 0.2
    np.testing.assert_allclose(geometry_outline(geometry)[:4],
                               [[0.9, 0.0], [0.9, 1.0], [1.1, 1.0], [1.1, 0.0]])
    geometry.geometry_type = 4
    arcs = geometry.arcs_of_circle
    arcs.r, arcs.z = [1.0, 2.0, 2.0, 1.0], [0.0, 0.0, 1.0, 1.0]
    arcs.curvature_radii = [0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(geometry_outline(geometry),
                               [[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]])
    arcs.curvature_radii = [1.0, 0.0, 0.0, 0.0]  # the bottom side bends
    bottom = geometry_outline(geometry)[:16]
    assert np.all(bottom[1:, 1] < 0)
    np.testing.assert_allclose(np.hypot(bottom[:, 0] - 1.5, bottom[:, 1] - np.sqrt(0.75)),
                               1.0)
    arcs.curvature_radii = [-1.0, 0.0, 0.0, 0.0]  # into the square
    assert np.all(geometry_outline(geometry)[1:16, 1] > 0)


@pytest.fixture(scope="module")
def two_beams(tmp_path_factory):
    """Two core sources with the same identifier, on grids of different lengths."""
    import imas

    path = tmp_path_factory.mktemp("beams") / "beams.nc"
    with imas.DBEntry(str(path), "w") as entry:
        ids = entry.factory.core_sources()
        ids.ids_properties.homogeneous_time = 1
        ids.time = [1.0]
        ids.source.resize(2)
        for k, source in enumerate(ids.source):
            source.identifier.name = "nbi"
            source.identifier.index = 2
            source.profiles_1d.resize(1)
            profiles = source.profiles_1d[0]
            profiles.time = 1.0
            rho = np.linspace(0.0, 1.0, 11 + 10 * k)
            profiles.grid.rho_tor_norm = rho
            profiles.grid.psi = 2.0 * rho**2
            profiles.electrons.energy = (k + 1) * (1.0 - rho**2)
        entry.put(ids)
    source = DataSource(parse_source(str(path)))
    yield source
    source.close()


def test_elements_with_the_same_name(two_beams):
    energy = "core_sources/source[:]/profiles_1d/electrons/energy"
    data = two_beams.signal(parse_signal_path(energy))
    assert data.names == ["nbi", "nbi#2"]
    # each has its own x values, also when they are given by a path
    psi_n = parse_signal_path("norm(core_sources/source[:]/profiles_1d/grid/psi)")
    data = two_beams.signal(parse_signal_path(energy), psi_n)
    assert [len(member.xs[0]) for member in data.members] == [11, 21]
    np.testing.assert_allclose(data.members[1].xs[0], np.linspace(0.0, 1.0, 21) ** 2)
