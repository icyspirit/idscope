import pytest

from idscope.data.paths import ALL, parse_alternatives, parse_signal_path
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


@pytest.mark.parametrize(
    "text, message",
    [
        ("sqrt(core_profiles/profiles_1d/grid/psi)", "Unknown function 'sqrt'"),
        ("norm(norm(core_profiles/profiles_1d/grid/psi))", "cannot be nested"),
        ("norm( )", "needs a signal path"),
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
    assert parse_signal_path("core_profiles/profiles_1d(itime)/zeff").indices[0] is ALL
    assert parse_signal_path("core_profiles/profiles_1d(:)/zeff").indices[0] is ALL


def test_dot_separators():
    path = parse_signal_path("core_profiles.profiles_1d[:].electrons.temperature[0]")
    assert path.parts == ("profiles_1d", "electrons", "temperature")
    assert path.indices == (ALL, None, 0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "core_profiles",
        "Core_profiles/time",
        "core_profiles/profiles_1d[1:3]/zeff",
        "core_profiles/profiles_1d(0)/zeff",
        "core_profiles/profiles_1d(i1)/zeff",
        "core_profiles/profiles_1d[0]/ion(1)/temperature",
        "core_profiles profiles_1d",
    ],
)
def test_invalid_paths(text):
    with pytest.raises(PathError):
        parse_signal_path(text)
