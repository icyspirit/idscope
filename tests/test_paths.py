import pytest

from idscope.data.paths import ALL, Range, parse_alternatives, parse_signal_path
from idscope.errors import PathError


def test_alternatives():
    paths = parse_alternatives(
        "equilibrium/time_slice/profiles_1d/j_phi | equilibrium/time_slice/profiles_1d/j_tor"
    )
    assert [p.parts[-1] for p in paths] == ["j_phi", "j_tor"]
    assert len(parse_alternatives("core_profiles/time")) == 1
    with pytest.raises(PathError, match="Empty alternative"):
        parse_alternatives("core_profiles/time |")


def test_norm_function():
    path = parse_signal_path(" norm( core_profiles/profiles_1d/grid/psi ) ")
    assert path.transform == "norm"
    assert path.dd_path == "profiles_1d/grid/psi"
    assert path.text == "norm( core_profiles/profiles_1d/grid/psi )"
    assert path.canonical() == "norm(core_profiles/profiles_1d/grid/psi)"
    assert path.label == path.name == "psi_n"
    assert parse_signal_path("core_profiles/profiles_1d/grid/psi").name == "psi"
    x = parse_alternatives(
        "core_profiles/profiles_1d/grid/rho_tor_norm | norm(core_profiles/profiles_1d/grid/psi)"
    )
    assert [p.transform for p in x] == [None, "norm"]


def test_transpose_function():
    path = parse_signal_path("transpose(equilibrium/time_slice/profiles_2d[0]/psi)")
    assert path.transform == "transpose" and path.transform_args == ()
    assert path.canonical() == "transpose(equilibrium/time_slice/profiles_2d[0]/psi)"
    assert path.name == "psi"
    assert path.label == "profiles_2d[0]/psi"
    # the axes, as numpy.transpose: listed, or in a tuple or list; commas inside
    # the index of the path are not arguments
    for text in ("transpose(bolometer/power_density/data[:, :, 0], 1, 0)",
                 "transpose( bolometer/power_density/data[:, :, 0] , (1, 0) )",
                 "transpose(bolometer/power_density/data[:, :, 0], [1, 0])"):
        path = parse_signal_path(text)
        assert path.data_index == (ALL, ALL, 0)
        assert path.transform_args == (1, 0)
        assert path.canonical() == "transpose(bolometer/power_density/data[:, :, 0], 1, 0)"


@pytest.mark.parametrize(
    "text, message",
    [
        ("sqrt(core_profiles/profiles_1d/grid/psi)",
         "Unknown function 'sqrt'.*available: norm, transpose"),
        ("norm(norm(core_profiles/profiles_1d/grid/psi))", "cannot be nested"),
        ("norm( )", "needs a signal path"),
        ("norm(core_profiles/profiles_1d/grid/psi, 1)", "takes a signal path only"),
        ("transpose(equilibrium/time_slice/profiles_2d[0]/psi, x)", "Invalid axis 'x'"),
        ("transpose(equilibrium/time_slice/profiles_2d[0]/psi, (1, 0)", "unbalanced"),
    ],
)
def test_invalid_functions(text, message):
    with pytest.raises(PathError, match=message):
        parse_signal_path(text)


def test_simple_path():
    path = parse_signal_path("core_profiles/profiles_1d/electrons/temperature")
    assert path.ids_name == "core_profiles"
    assert path.occurrence == 0
    assert path.parts == ("profiles_1d", "electrons", "temperature")
    assert path.indices == (None, None, None)
    assert path.dd_path == "profiles_1d/electrons/temperature"
    assert path.label == "electrons/temperature"


def test_occurrence_and_python_indices():
    path = parse_signal_path(" equilibrium:2/time_slice[:]/profiles_2d[1]/psi ")
    assert path.ids_name == "equilibrium"
    assert path.occurrence == 2
    assert path.indices == (ALL, 1, None)
    assert path.text == "equilibrium:2/time_slice[:]/profiles_2d[1]/psi"
    assert path.canonical() == "equilibrium:2/time_slice[:]/profiles_2d[1]/psi"
    assert path.label == "profiles_2d[1]/psi"


def test_fortran_indices_are_one_based():
    path = parse_signal_path("magnetics/flux_loop(4)/flux/data")
    assert path.indices == (3, None, None)
    assert path.canonical() == "magnetics/flux_loop[3]/flux/data"
    assert path.label == "flux_loop[3]/flux/data"
    # (itime) stands for the time axis, as in the Data Dictionary documentation
    assert parse_signal_path("core_profiles/profiles_1d(itime)/zeff").indices[0] is None
    assert parse_signal_path("core_profiles/profiles_1d(:)/zeff").indices[0] == ALL
    # Fortran ranges include both ends
    assert parse_signal_path("pf_active/coil(1:6)/current/data").indices[0] == Range(0, 6)


def test_dot_separators():
    path = parse_signal_path("core_profiles.profiles_1d[:].electrons.temperature[0]")
    assert path.parts == ("profiles_1d", "electrons", "temperature")
    assert path.indices == (ALL, None, None)
    assert path.data_index == (0,)


def test_wildcards_ranges_and_numpy_index():
    path = parse_signal_path("pf_active/coil[:]/current/data[0]")
    assert path.indices == (ALL, None, None)
    assert path.data_index == (0,)
    assert path.canonical() == "pf_active/coil[:]/current/data[0]"
    assert path.label == "current/data[0]"
    path = parse_signal_path("equilibrium/time_slice/profiles_2d[0]/psi[:, 40]")
    assert path.indices == (None, 0, None)
    assert path.data_index == (ALL, 40)
    assert path.canonical() == "equilibrium/time_slice/profiles_2d[0]/psi[:, 40]"
    path = parse_signal_path("pf_active.coil[-1].current.data[0:10:2]")
    assert path.indices == (-1, None, None)
    assert path.data_index == (Range(0, 10, 2),)
    assert path.canonical() == "pf_active/coil[-1]/current/data[0:10:2]"
    assert parse_signal_path("pf_active/coil[6:]/current/data").indices[0] == Range(6)
    path = parse_signal_path("equilibrium/time_slice/boundary/x_point[:]")
    assert path.indices == (None, None, None) and path.data_index == (ALL,)


@pytest.mark.parametrize(
    "text, message",
    [
        ("", "expected 'ids_name/path/to/quantity'"),
        ("core_profiles", "only names the IDS"),
        ("Core_profiles/time", "expected"),
        ("core_profiles/profiles_1d(0)/zeff", "start at 1"),
        ("core_profiles/profiles_1d[0]/ion(1)/temperature", "cannot be mixed"),
        ("core_profiles profiles_1d", "expected"),
        ("pf_active/coil[1, 2]/current/data", "takes one index"),
        ("pf_active/coil[a]/current/data", "expected a number"),
        ("pf_active/coil[1:2:0]/current/data", "the step is 0"),
        ("pf_active/coil[]/current/data", "Empty index"),
        ("pf_active/coil(1]/current/data", "unbalanced"),
        ("pf_active/coil[0]/current/data[itime, 0]", "placeholders"),
        ("pf_active/coil[0]/current/data]", "unexpected"),
        ("pf_active/coil[*]/current/data", "write ':' for all elements"),
    ],
)
def test_invalid_paths(text, message):
    with pytest.raises(PathError, match=message):
        parse_signal_path(text)
