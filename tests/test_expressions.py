import numpy as np
import pytest

from idscope.data import (
    Axis,
    Channels,
    MapSeries,
    ProfileSeries,
    SignalPath,
    Trace,
    parse_alternatives,
    parse_signal_path,
)
from idscope.data.expressions import (
    Expression,
    combine,
    derivative,
    parse_signal,
    relative_to,
)
from idscope.errors import NoDataError, PathError

IP = "equilibrium/time_slice/global_quantities/ip"
SUMMARY_IP = "summary/global_quantities/ip/value"
TE = "core_profiles/profiles_1d/electrons/temperature"
TI = "core_profiles/profiles_1d/t_i_average"
TIME = np.linspace(0.0, 10.0, 41)
RAMP = np.minimum(TIME / 2.0, 1.0)


def value(source, text):
    (signal,) = parse_alternatives(text)
    return source.signal(signal)


# ---------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    IP, "magnetics/flux_loop(2)/flux/data", f"{TE}[-1]", "equilibrium:1/time_slice/q",
    "norm(equilibrium/time_slice/profiles_1d/psi)",
    "transpose(equilibrium/time_slice/profiles_2d[0]/psi, 1, 0)",
    "core_profiles/profiles_1d/ electrons/temperature",  # spaces, as before
])
def test_paths_stay_paths(text):
    parsed = parse_signal(text)
    assert isinstance(parsed, SignalPath)
    assert parsed == parse_signal_path(text)


def test_expressions():
    difference = parse_signal(f"{IP} - {SUMMARY_IP}")
    assert isinstance(difference, Expression)
    assert difference.label == "global_quantities/ip - ip/value"
    assert [path.text for path in difference.paths()] == [IP, SUMMARY_IP]
    # "/" between nodes is a path; with spaces around it, a division
    assert isinstance(parse_signal("a/b"), SignalPath)
    ratio = parse_signal(f"{TI} / {TE}")
    assert ratio.label == "profiles_1d/t_i_average / electrons/temperature"
    # precedence, parentheses and unary minus, in the labels
    assert parse_signal(f"{TE} - {TI} * 2").label == \
        "electrons/temperature - profiles_1d/t_i_average * 2"
    assert parse_signal(f"({TE} - {TI}) / {TE}").label == \
        "(electrons/temperature - profiles_1d/t_i_average) / electrons/temperature"
    assert parse_signal(f"{TE} - ({TI} - {TE})").label == \
        "electrons/temperature - (profiles_1d/t_i_average - electrons/temperature)"
    assert parse_signal(f"-({IP} - {SUMMARY_IP})").label == \
        "-(global_quantities/ip - ip/value)"
    assert parse_signal(f"deriv({IP}) * 1e-6").label == "deriv(global_quantities/ip) * 1e-6"
    # norm() of a path is still a path; of an expression, an expression
    assert isinstance(parse_signal(f"norm({TE}) * 2"), Expression)
    assert isinstance(parse_signal(f"norm({TE} / {TI})"), Expression)
    # an index with a minus sign is not a subtraction
    assert parse_signal(f"{TE}[-1] + 1").label == "electrons/temperature[-1] + 1"
    # the same expression, however written, is read once by a source
    assert parse_signal(f"{IP}-{SUMMARY_IP}").canonical() == difference.canonical()
    # alternatives of expressions
    first, second = parse_alternatives(f"{IP} * 2 | {SUMMARY_IP}")
    assert isinstance(first, Expression) and isinstance(second, SignalPath)


@pytest.mark.parametrize("text, message", [
    (f"{TE} +", "expected a signal path, a number or '\\(' at the end"),
    (f"deriv({TE}", "unbalanced brackets"),
    (f"sqrt({TE})", "unknown function 'sqrt' \\(available: deriv, abs, norm, transpose\\)"),
    (f"transpose({TE} * 2)", "transpose\\(\\) takes a signal path, not an expression"),
    (f"{TE} ** 2", "expected a signal path"),
    (f"{TE} - core_profiles", "only names the IDS"),
])
def test_invalid_expressions(text, message):
    with pytest.raises(PathError, match=message):
        parse_signal(text)


def test_a_number_is_not_a_signal(source_a):
    with pytest.raises(PathError, match="is a number, not a signal"):
        value(source_a, "2 * 3")


# ---------------------------------------------------------------------------------
# Evaluation by a data source
# ---------------------------------------------------------------------------------


def test_traces(source_a):
    difference = value(source_a, f"{IP} - {SUMMARY_IP}")
    assert isinstance(difference, Trace)
    np.testing.assert_allclose(difference.values, -2.0e6 * RAMP)
    assert difference.value_axis == Axis("global_quantities/ip - ip/value", "A")
    assert difference.path.text == f"{IP} - {SUMMARY_IP}"
    scaled = value(source_a, f"-{IP} * 1e-6")
    np.testing.assert_allclose(scaled.values, RAMP)
    assert scaled.value_axis.units == "A"  # scaled by a number: the same quantity
    rate = value(source_a, f"deriv({SUMMARY_IP})")
    assert rate.value_axis == Axis("deriv(ip/value)", "A/s")
    np.testing.assert_allclose(rate.values[:7], 0.5e6)
    assert value(source_a, f"abs({IP})").values.min() >= 0
    ratio = value(source_a, f"{SUMMARY_IP} / {SUMMARY_IP}")
    assert ratio.value_axis.units == ""  # dimensionless
    product = value(source_a, f"{SUMMARY_IP} * {IP}")
    assert product.value_axis.units == "A.A"
    # read once, then from the cache of the source
    assert value(source_a, f"{IP}-{SUMMARY_IP}") is difference


def test_profiles(source_a):
    ratio = value(source_a, f"{TI} / {TE}")
    assert isinstance(ratio, ProfileSeries) and len(ratio.ys) == 41
    np.testing.assert_allclose(ratio.ys[20][:10], 0.8)
    assert ratio.value_axis.units == ""
    # divided by a time trace: the central value, at the time of each profile
    peaked = value(source_a, f"{TE} / {TE}[0]")
    np.testing.assert_allclose([y[0] for y in peaked.ys], 1.0)
    slope = value(source_a, f"deriv({TE})")
    x, y, _ = slope.at(5.0)
    _, te, _ = value(source_a, TE).at(5.0)
    np.testing.assert_allclose(y, np.gradient(te, x))
    assert slope.value_axis.units == "eV"  # rho_tor_norm has no units
    normalized = value(source_a, f"norm({TE} * 2)")
    np.testing.assert_allclose([y[0] for y in normalized.ys], 0.0)
    with pytest.raises(PathError, match="profiles against rho_tor_norm and psi"):
        value(source_a, f"{TE} - equilibrium/time_slice/profiles_1d/q")


def test_maps_and_members(source_a):
    zero = value(source_a, "equilibrium/time_slice/profiles_2d[0]/psi * 2 - "
                 "equilibrium/time_slice/profiles_2d[0]/psi * 2")
    assert isinstance(zero, MapSeries) and zero.count == 41
    x1, x2, z, _ = zero.at(5.0)
    assert z.shape == (len(x1), len(x2)) and np.all(z == 0)
    currents = value(source_a, "abs(pf_active/coil[:]/current/data) * 1e-3")
    assert isinstance(currents, Channels) and currents.names[6] == "PF1"
    np.testing.assert_allclose(currents.members[6].values, 4.0 * RAMP)
    # member by member, and with one signal for all
    zero = value(source_a, "pf_active/coil[:]/current/data - pf_active/coil[:]/current/data")
    assert all(np.all(member.values == 0) for member in zero.members)
    minus_ip = value(source_a, f"pf_active/coil[:]/current/data - {SUMMARY_IP}")
    np.testing.assert_allclose(minus_ip.members[6].values, (-4.0e3 - 1.0e6) * RAMP)
    with pytest.raises(PathError, match="applies to time traces and profiles"):
        value(source_a, "deriv(equilibrium/time_slice/profiles_2d[0]/psi)")
    with pytest.raises(PathError, match="shapes cannot be computed with"):
        value(source_a, "wall/description_2d[0]/limiter/unit[:]/outline * 2")


def test_what_a_source_does_not_have(source_a):
    """Not filled in one of the signals: not filled (a note, not an error)."""
    with pytest.raises(NoDataError):
        value(source_a, f"{IP} - summary/global_quantities/beta_pol/value")
    # the first alternative that can be read
    data = source_a.first_signal(parse_alternatives(
        f"{IP} - summary/global_quantities/beta_pol/value | {IP} * 2"))
    np.testing.assert_allclose(data.values, -2.0e6 * RAMP)


# ---------------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------------

A = parse_signal_path("summary/global_quantities/ip/value")


def trace(time, values, units="A"):
    return Trace(A, None if time is None else np.asarray(time, float),
                 np.asarray(values, float), Axis("ip", units))


def profiles(time, xs, ys, x_label="rho_tor_norm"):
    return ProfileSeries(A, None if time is None else np.asarray(time, float), xs, ys,
                         Axis(x_label, ""), Axis("te", "eV"))


def test_traces_on_the_times_of_the_first():
    a = trace(np.arange(0.0, 11.0), np.arange(0.0, 11.0))
    b = trace(np.arange(0.0, 5.5, 0.5), 2.0 * np.arange(0.0, 5.5, 0.5))
    difference = combine("-", a, b)
    np.testing.assert_array_equal(difference.time, a.time)
    np.testing.assert_allclose(difference.values[:6], -np.arange(0.0, 6.0))
    assert np.isnan(difference.values[6:]).all()  # where the second has no data
    # a constant, and a single sample (only at its time)
    np.testing.assert_allclose(combine("+", a, trace(None, [1.0])).values, a.values + 1)
    alone = combine("-", a, trace([3.0], [1.0]))
    assert np.isnan(alone.values[np.arange(11) != 3]).all() and alone.values[3] == 2.0
    # the second without time, the first with: on the times of the first
    assert combine("*", trace(None, [2.0]), a).time is a.time


def test_profiles_on_the_grid_and_times_of_the_first():
    grid, coarse = np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 3)
    a = profiles([1.0, 2.0, 3.0], [grid] * 3, [grid * 0 + k for k in (1.0, 2.0, 3.0)])
    b = profiles([2.0, 2.8], [coarse] * 2, [coarse, 2 * coarse])
    difference = combine("-", a, b)
    assert len(difference.ys) == 3 and difference.xs[0] is grid
    # before the second (by more than half its step): none of it
    assert np.isnan(difference.ys[0]).all()
    np.testing.assert_allclose(difference.ys[1], 2.0 - grid)  # interpolated
    np.testing.assert_allclose(difference.ys[2], 3.0 - 2 * grid)  # its closest slice
    # a static profile times a time trace: a profile at each time of the trace
    scaled = combine("*", profiles(None, [grid], [grid]), trace([0.0, 1.0], [1.0, 2.0]))
    np.testing.assert_array_equal(scaled.time, [0.0, 1.0])
    np.testing.assert_allclose(scaled.ys[1], 2 * grid)
    with pytest.raises(PathError, match="against rho_tor_norm and psi_n"):
        combine("-", a, profiles(None, [grid], [grid], x_label="psi_n"))


def test_maps_on_the_grid_of_the_first():
    x1, x2 = np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 9)
    fine = np.linspace(0.0, 1.0, 11)
    plane = MapSeries(A, None, 1, lambda i: (x1, x2, np.add.outer(x1, x2)),
                      Axis("R", "m"), Axis("Z", "m"), Axis("psi", "Wb"))
    other = MapSeries(A, None, 1, lambda i: (fine, x2, np.add.outer(fine, x2)),
                      Axis("R", "m"), Axis("Z", "m"), Axis("psi", "Wb"))
    _, _, z = combine("-", plane, other).at_index(0)
    np.testing.assert_allclose(z, 0.0, atol=1e-12)
    ramp = trace([0.0, 1.0], [1.0, 3.0])
    scaled = combine("*", plane, ramp)  # a static map at the times of a trace
    assert scaled.count == 2
    np.testing.assert_allclose(scaled.at_index(1)[2], 3 * np.add.outer(x1, x2))
    with pytest.raises(PathError, match="a map combines with maps, numbers and time"):
        combine("-", plane, profiles(None, [x1], [x1]))


def test_members_by_name():
    def members(names, values):
        data = [trace(None, [v]) for v in values]
        return Channels(A, data, names, Axis("current", "A"), axis_label="coil")

    a = members(["PF1", "PF2", "PF3"], [1.0, 2.0, 3.0])
    by_name = combine("-", a, members(["PF3", "PF1"], [30.0, 10.0]))
    assert by_name.names == ["PF1", "PF3"]  # those both have
    np.testing.assert_allclose([float(m.values) for m in by_name.members], [-9.0, -27.0])
    renamed = combine("-", a, members(["P1", "P2", "P3"], [1.0, 1.0, 1.0]))
    assert renamed.names == a.names  # all renamed: in order
    with pytest.raises(PathError, match="no element of the same name"):
        combine("-", a, members(["X"], [1.0]))


def test_derivatives():
    t = np.array([0.0, 1.0, 1.0, 3.0, np.nan])
    rate = derivative(trace(t, [0.0, 2.0, 2.0, 6.0, 1.0]))
    np.testing.assert_allclose(rate.values[[0, 1, 3]], 2.0)  # the same time once
    assert np.isnan(rate.values[4]) and rate.value_axis.units == "A/s"
    with pytest.raises(PathError, match="has one value"):
        derivative(trace(None, [1.0]))
    x = np.linspace(0.0, 1.0, 21)
    slope = derivative(profiles(None, [x[::-1]], [x[::-1] ** 2]))  # x decreasing
    np.testing.assert_allclose(slope.ys[0][1:-1], 2 * x[::-1][1:-1], rtol=1e-12)


def test_relative_to_a_reference():
    a = trace([0.0, 1.0, 2.0], [1.0, 2.0, 4.0])
    b = trace([0.0, 2.0], [2.0, 8.0])
    difference = relative_to(b, a, "difference")
    np.testing.assert_allclose(difference.values, [1.0, 4.0])  # on the times of B
    assert difference.value_axis == Axis("Δ ip", "A")
    ratio = relative_to(b, a, "ratio")
    np.testing.assert_allclose(ratio.values, [2.0, 2.0])
    assert ratio.value_axis == Axis("ip ratio", "")
