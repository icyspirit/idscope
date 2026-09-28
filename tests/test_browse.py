from idscope.data.browse import find_item, list_signals


def test_core_profiles_items(source_a):
    items = list_signals(source_a, "core_profiles")
    te = find_item(items, "profiles_1d/electrons/temperature")
    assert te.kind == "profile" and te.time_dependent
    assert te.signal_path == "core_profiles/profiles_1d/electrons/temperature"
    assert te.units == "eV"
    assert te.dimensions == ("rho_tor_norm",)  # the order an index selects them in
    ion = find_item(items, "profiles_1d/ion/temperature")
    assert ion.signal_path == "core_profiles/profiles_1d/ion[:]/temperature"
    assert find_item(items, "global_quantities/ip").kind == "trace"
    assert find_item(items, "time") is None
    assert find_item(items, "ids_properties/homogeneous_time") is None
    assert find_item(list_signals(source_a, "core_profiles", include_all=True), "time")


def test_magnetics_and_equilibrium_items(source_a):
    flux = find_item(list_signals(source_a, "magnetics"), "flux_loop/flux/data")
    assert flux.kind == "trace"
    assert flux.signal_path == "magnetics/flux_loop[:]/flux/data"
    eq = list_signals(source_a, "equilibrium")
    psi = find_item(eq, "time_slice/profiles_2d/psi")
    assert psi.kind == "map"
    assert psi.signal_path == "equilibrium/time_slice/profiles_2d[0]/psi"
    assert psi.dimensions == ("grid/dim1", "grid/dim2")
    assert flux.dimensions == ("time",)
    assert find_item(eq, "vacuum_toroidal_field/r0").kind == "constant"


def test_shape_items(source_a):
    coils = find_item(list_signals(source_a, "pf_active"), "coil/element/geometry")
    assert coils.kind == "shape" and not coils.time_dependent
    assert coils.signal_path == "pf_active/coil[:]/element[:]/geometry"
    eq = list_signals(source_a, "equilibrium")
    boundary = find_item(eq, "time_slice/boundary/outline")
    assert boundary.kind == "shape" and boundary.time_dependent
    assert find_item(eq, "time_slice/global_quantities/magnetic_axis").kind == "shape"
    wall = find_item(list_signals(source_a, "wall"), "description_2d/limiter/unit/outline")
    assert wall.signal_path == "wall/description_2d[:]/limiter/unit[:]/outline"
