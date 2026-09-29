import pytest

from idscope.data import parse_signal_path
from idscope.data.sources import (
    DataSource,
    SourcePool,
    SourceSpec,
    format_source_list,
    parse_source,
    parse_source_list,
)
from idscope.errors import NoDataError, PathError, SourceError


def test_parse_source_labels():
    assert parse_source("A=/data/a.nc") == SourceSpec("/data/a.nc", "A")
    assert parse_source("/data/shot_12.nc").label == "shot_12"
    assert parse_source("imas:hdf5?path=/data/134173/106/").label == "106"
    uri = "imas:mdsplus?user=public;pulse=134173;run=106;database=ITER;version=3"
    assert parse_source(uri) == SourceSpec(uri, "134173/106")
    assert parse_source(f"ref = {uri}") == SourceSpec(uri, "ref")


def test_parse_source_list():
    specs = parse_source_list("a.nc, b.nc,, a.nc")
    assert [s.label for s in specs] == ["a", "b", "a#2"]
    text = format_source_list(parse_source_list("x=a.nc, b.nc"))
    assert text == "x=a.nc, b.nc"
    with pytest.raises(SourceError):
        parse_source("  ")


def test_format_relative_to_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "configs"
    specs = parse_source_list(f"x={base / 'a.nc'}, b.nc, imas:hdf5?path=e")
    # inside the directory: relative; outside: absolute, to read it back from there
    assert format_source_list(specs, base) == f"x=a.nc, {tmp_path / 'b.nc'}, imas:hdf5?path=e"
    assert parse_source_list(format_source_list(specs, base), base)[1].uri == str(
        tmp_path / "b.nc"
    )


def test_hdf5_directory_becomes_uri(data_dir):
    spec = parse_source(str(data_dir / "h5"))
    assert spec.uri == f"imas:hdf5?path={(data_dir / 'h5').resolve()}"


def test_available_ids(source_a):
    available = source_a.available_ids()
    for name in ("core_profiles", "equilibrium", "magnetics", "summary"):
        assert (name, 0) in available


def test_signal_cache(source_a):
    path = parse_signal_path("core_profiles/global_quantities/ip")
    assert source_a.signal(path) is source_a.signal(path)
    bad = parse_signal_path("core_profiles/global_quantities/ipp")
    for _ in range(2):
        with pytest.raises(PathError):
            source_a.signal(bad)
    source_a.close()
    assert source_a.signal(path).values.shape == (41,)


def test_first_signal_alternatives(source_a, dd3_file):
    from idscope.data import parse_alternatives

    j = parse_alternatives(
        "equilibrium/time_slice/profiles_1d/j_phi | equilibrium/time_slice/profiles_1d/j_tor"
    )
    assert source_a.first_signal(j).path.parts[-1] == "j_phi"  # DD 4
    dd3 = DataSource(parse_source(str(dd3_file)))
    try:
        assert dd3.first_signal(j).path.parts[-1] == "j_tor"  # DD 3
    finally:
        dd3.close()
    # j_tor does not exist in DD 4, j_phi exists but is not filled: report the latter
    missing = parse_alternatives(
        "core_profiles/profiles_1d/j_phi | core_profiles/profiles_1d/j_tor"
    )
    with pytest.raises(NoDataError, match=r"j_phi' is not filled\. \(tried"):
        source_a.first_signal(missing)


def test_first_signal_x_alternatives(source_a):
    from idscope.data import parse_alternatives

    q = parse_alternatives("equilibrium/time_slice/profiles_1d/q")
    x = parse_alternatives(
        "equilibrium/time_slice/profiles_1d/psi_norm"
        " | norm(equilibrium/time_slice/profiles_1d/psi)"
    )
    assert source_a.first_signal(q, x).x_axis.title() == "psi_n"
    # a quantity that is not filled is reported as such, whatever its x
    gm1 = parse_alternatives("equilibrium/time_slice/profiles_1d/gm1")
    with pytest.raises(NoDataError) as info:
        source_a.first_signal(gm1, x)
    assert str(info.value) == "'equilibrium/time_slice/profiles_1d/gm1' is not filled."
    # x paths that cannot be read are listed
    x_missing = parse_alternatives(
        "equilibrium/time_slice/profiles_1d/psi_norm"
        " | equilibrium/time_slice/profiles_1d/rho_tor"
    )
    with pytest.raises(NoDataError, match=r"psi_norm' is not filled\. \(tried x: "):
        source_a.first_signal(q, x_missing)


def test_missing_file(tmp_path):
    source = DataSource(parse_source(str(tmp_path / "missing.nc")))
    with pytest.raises((SourceError, NoDataError)):
        source.signal(parse_signal_path("core_profiles/global_quantities/ip"))


def test_pool(data_dir):
    pool = SourcePool()
    a = pool.get(parse_source(f"A={data_dir / 'a.nc'}"))
    assert pool.get(parse_source(f"B={data_dir / 'a.nc'}")) is a
    assert a.label == "B"
    pool.keep_only([])
    assert pool.sources() == []
    pool.close()


@pytest.mark.parametrize("name", ["a.nc", "h5"])
def test_filled_paths_read_when_they_cannot_be_listed(data_dir, name, old_access_layer):
    source = DataSource(parse_source(str(data_dir / name)))
    for ids_name in ("core_profiles", "equilibrium", "pf_active", "wall"):
        listed = old_access_layer(source.entry, ids_name)  # the real listing
        assert source.filled_paths(ids_name) == sorted(listed)
    assert "Access Layer version 5.7" in source.listing_notes[("core_profiles", 0)]
    assert sum("cannot list its filled quantities" in note for note in source.notes) == 1
    source.close()


def test_read_filled_paths_samples_long_arrays():
    import imas

    from idscope.data.sources import _spread, read_filled_paths

    assert _spread(3, 16) == [0, 1, 2]
    assert _spread(41, 3) == [0, 20, 40]
    assert len(_spread(100, 16)) == 16 and _spread(100, 16)[-1] == 99

    cp = imas.IDSFactory().core_profiles()
    cp.ids_properties.homogeneous_time = 1
    cp.time = [0.0, 1.0, 2.0, 3.0, 4.0]
    cp.profiles_1d.resize(5)
    cp.profiles_1d[1].zeff = [1.5]  # a time slice that is not read
    cp.profiles_1d[2].ion.resize(3)
    cp.profiles_1d[2].ion[1].temperature = [1.0]  # every ion is read
    assert read_filled_paths(cp) == [
        "ids_properties/homogeneous_time", "profiles_1d/ion/temperature", "time",
    ]
