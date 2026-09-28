import numpy as np
import pytest

from idscope.data import (
    MapSeries,
    ProfileSeries,
    Trace,
    parse_signal_path,
)
from idscope.data.extract import normalize
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
        ("pf_active/coil[0]/current/data", "pf_active"),
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
