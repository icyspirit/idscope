"""Values of GGD quantities on unstructured grids, as maps of their cells."""

import numpy as np
import pytest

from idscope.data import Axis, MapSeries, MeshSeries, ProfileSeries, Trace, parse_signal_path
from idscope.data.expressions import combine, derivative, parse_signal, relative_to
from idscope.data.extract import Mesh, _children, _grid_mesh, extract_signal
from idscope.demo import EDGE_GRID, EDGE_TIMES
from idscope.errors import PathError

TE = "edge_profiles/ggd/electrons/temperature[0]/values"
NE = "edge_profiles/ggd/electrons/density[0]/values"
N_THETA, N_RADIAL = EDGE_GRID


def get(source, text):
    return source.signal(parse_signal_path(text))


@pytest.fixture(scope="module")
def dd3_edge(tmp_path_factory):
    """The demo edge_profiles in Data Dictionary 3.39.0."""
    import imas

    from idscope.demo import edge_profiles

    path = tmp_path_factory.mktemp("edge") / "edge.nc"
    with imas.DBEntry(str(path), "w", dd_version="3.39.0") as entry:
        entry.put(edge_profiles(entry.factory))
    return path


def test_values_on_cells_and_nodes(source_a):
    te = get(source_a, TE)
    assert isinstance(te, MeshSeries) and te.on == "cells" and te.kind == "map"
    np.testing.assert_allclose(te.time, EDGE_TIMES)
    assert te.value_axis.units == "eV"
    assert (te.x_axis.title(), te.x2_axis.title()) == ("R [m]", "Z [m]")
    mesh, values, t = te.at(6.0)
    assert t == 6.0 and len(mesh.cells) == N_THETA * N_RADIAL
    assert len(mesh.points) == N_THETA * (N_RADIAL + 1)
    # the outboard cell of the innermost ring: the hottest
    assert values[0] == pytest.approx(np.nanmax(values))
    assert values[0] == pytest.approx(  # at 6 s
        100.0 * 0.8 * (1 + 0.4 * np.cos(np.pi / N_THETA)) * np.exp(-0.5 / N_RADIAL / 0.3))
    ne = get(source_a, NE)
    assert ne.on == "nodes" and ne.value_axis.units == "m^-3"
    mesh, cells, _ = ne.at(6.0)
    # on the cells: the mean of their nodes
    nodes = get(source_a, "edge_profiles/ggd[2]/electrons/density[0]/values")
    assert isinstance(nodes, MeshSeries) and nodes.count == 1  # one time slice
    first = mesh.cells[0]
    assert cells[0] == pytest.approx(np.mean(_node_values(source_a)[first]))


def _node_values(source):
    ids = source.get_ids("edge_profiles")
    return np.asarray(ids.ggd[2].electrons.density[0].values)


def test_the_cells_of_a_point(source_a):
    mesh, values, _ = get(source_a, TE).at(2.0)
    cell = mesh.cell_at(2.3 + 0.01, 0.001)  # just outside the boundary, outboard
    assert cell == 0
    assert mesh.cell_at(1.8, 0.0) is None  # the plasma: no cell
    assert mesh.cell_at(3.0, 0.0) is None  # beyond the wall


def test_data_dictionary_3(dd3_edge):
    from idscope.data import DataSource, parse_source

    source = DataSource(parse_source(str(dd3_edge)))
    te = get(source, TE)
    mesh, values, _ = te.at(6.0)
    assert te.on == "cells" and len(mesh.cells) == N_THETA * N_RADIAL
    np.testing.assert_allclose(mesh.points[0], [1.8 + 0.5, 0.0])  # R first, as said
    source.close()


def test_read_at_once_as_one_by_one(source_a):
    """From a netCDF file the grid is read at once: the same as one by one."""
    import imas

    lazy = source_a.get_ids("edge_profiles").grid_ggd[0].space[0]
    with imas.DBEntry(str(source_a.spec.uri), "r") as entry:
        eager = entry.get("edge_profiles").grid_ggd[0].space[0]
    for dimension, child in ((0, "geometry"), (2, "nodes")):
        at_once = _children(lazy.objects_per_dimension[dimension].object, child)
        one_by_one = _children(eager.objects_per_dimension[dimension].object, child)
        assert len(at_once) == len(one_by_one)
        for a, b in zip(at_once, one_by_one):
            np.testing.assert_array_equal(a, b)
    subset = source_a.get_ids("edge_profiles").grid_ggd[0].grid_subset[1]
    indices = _children(subset.element, "object/index")
    assert len(indices) == N_THETA * N_RADIAL and indices[5].tolist() == [6.0]


# ---------------------------------------------------------------------------------
# Grids written in other ways
# ---------------------------------------------------------------------------------


def edge(nodes, cells, *, z_first=False, subsets=(), values=(), times=(0.0,)):
    """An edge_profiles in memory: a grid of ``nodes`` (R, Z) and ``cells`` (node
    indices from 1), and quantities ``values`` as (grid subset, values)."""
    import imas
    from imas.ids_identifiers import identifiers

    ids = imas.IDSFactory().edge_profiles()
    ids.ids_properties.homogeneous_time = 1
    ids.time = np.array(times, dtype=float)
    ids.grid_ggd.resize(len(times))
    for grid, t in zip(ids.grid_ggd, times):
        grid.time = t
        grid.space.resize(1)
        space = grid.space[0]
        names = ("z", "r") if z_first else ("r", "z")
        space.coordinates_type.resize(2)
        for kind, name in zip(space.coordinates_type, names):
            member = identifiers.coordinate_identifier[name]
            kind.name, kind.index = member.name, member.index
        space.objects_per_dimension.resize(3)
        space.objects_per_dimension[0].object.resize(len(nodes))
        for node, point in zip(space.objects_per_dimension[0].object, nodes):
            node.geometry = point[::-1] if z_first else point
        space.objects_per_dimension[2].object.resize(len(cells))
        for cell, indices in zip(space.objects_per_dimension[2].object, cells):
            cell.nodes = np.array(indices, dtype=np.int32)
        grid.grid_subset.resize(len(subsets))
        for subset, (index, dimension, objects) in zip(grid.grid_subset, subsets):
            subset.identifier.index = index
            subset.dimension = dimension
            subset.element.resize(len(objects))
            for element, number in zip(subset.element, objects):
                element.object.resize(1)
                element.object[0].space = 1
                element.object[0].dimension = dimension
                element.object[0].index = number
    ids.ggd.resize(len(times))
    for ggd, t in zip(ids.ggd, times):
        ggd.time = t
        ggd.electrons.temperature.resize(len(values))
        for scalar, (subset_index, data) in zip(ggd.electrons.temperature, values):
            scalar.grid_subset_index = subset_index
            scalar.values = np.asarray(data, dtype=float) * (1 + t)
    return ids


SQUARES = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
"""Two unit squares side by side, and a triangle on top of the right one."""


def test_cells_of_any_order_and_shape():
    nodes = SQUARES + [(1.5, 2.0)]
    # the first square row by row (not around it), the second around it
    ids = edge(nodes, [[1, 2, 4, 5], [2, 3, 6, 5], [5, 6, 7]], values=[(5, [1.0, 2.0, 3.0])])
    data = extract_signal(ids, parse_signal_path("edge_profiles/ggd/electrons/"
                                                 "temperature[0]/values"))
    mesh, values, _ = data.at(0.0)
    assert mesh.cells[0].tolist() == [0, 1, 4, 3]  # put in order around it
    assert mesh.cells[1].tolist() == [1, 2, 5, 4]  # kept
    assert len(mesh.cells[2]) == 3
    np.testing.assert_allclose(values, [1.0, 2.0, 3.0])
    assert mesh.cell_at(0.5, 0.5) == 0 and mesh.cell_at(1.5, 1.2) == 2


def test_coordinates_z_first_and_subsets():
    # the nodes of the space give Z first; a subset of one cell, element by element
    ids = edge(SQUARES, [[1, 2, 5, 4], [2, 3, 6, 5]], z_first=True,
               subsets=[(20, 3, [2])], values=[(20, [7.0]), (1, np.arange(6.0)),
                                               (2, [1.0, 2.0])])
    temperature = "edge_profiles/ggd/electrons/temperature[{}]/values"
    one = extract_signal(ids, parse_signal_path(temperature.format(0)))
    mesh, values, _ = one.at(0.0)
    np.testing.assert_allclose(mesh.points[2], [2.0, 0.0])  # R, Z
    assert np.isnan(values[0]) and values[1] == 7.0
    # on the nodes: the mean over the nodes of each cell
    nodes = extract_signal(ids, parse_signal_path(temperature.format(1)))
    np.testing.assert_allclose(nodes.at(0.0)[1], [np.mean([0, 1, 4, 3]),
                                                  np.mean([1, 2, 5, 4])])
    # on the edges: not a map, the values along their index as before
    edges = extract_signal(ids, parse_signal_path(temperature.format(2)))
    assert isinstance(edges, ProfileSeries)


def test_the_grid_at_the_time_of_the_values():
    """With a grid per time slice, the values of a slice are on its own grid."""
    ids = edge(SQUARES, [[1, 2, 5, 4], [2, 3, 6, 5]], values=[(5, [1.0, 2.0])],
               times=(0.0, 1.0))
    ids.grid_ggd[1].space[0].objects_per_dimension[0].object[0].geometry = [-1.0, 0.0]
    data = extract_signal(ids, parse_signal_path("edge_profiles/ggd/electrons/"
                                                 "temperature[0]/values"))
    assert data.count == 2
    first, _, _ = data.at(0.0)
    second, values, _ = data.at(1.0)
    assert first.points[0, 0] == 0.0 and second.points[0, 0] == -1.0
    np.testing.assert_allclose(values, [2.0, 4.0])
    assert _grid_mesh(ids.grid_ggd[1]) is second  # read once


def test_values_that_do_not_fit_are_a_profile():
    ids = edge(SQUARES, [[1, 2, 5, 4], [2, 3, 6, 5]], values=[(5, [1.0, 2.0, 3.0])])
    data = extract_signal(ids, parse_signal_path("edge_profiles/ggd/electrons/"
                                                 "temperature[0]/values"))
    assert isinstance(data, ProfileSeries)


# ---------------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------------

TWO_CELLS = Mesh(SQUARES, [[0, 1, 4, 3], [1, 2, 5, 4]])


def on_cells(time, values, mesh=TWO_CELLS):
    """Values on the cells of ``mesh`` (or of a mesh per slice): a row per slice."""
    values = np.atleast_2d(np.asarray(values, dtype=float))
    meshes = mesh if isinstance(mesh, list) else [mesh] * len(values)
    return MeshSeries(parse_signal_path(TE), None if time is None else np.asarray(time),
                      len(values), lambda i: (meshes[i], values[i]), Axis("te", "eV"))


def test_arithmetic_on_cells():
    a = on_cells([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])
    b = on_cells([0.0, 1.0], [[1.0, 1.0], [2.0, 2.0]])
    mesh, values = combine("-", a, b).at_index(1)
    assert mesh is TWO_CELLS
    np.testing.assert_allclose(values, [1.0, 2.0])  # cell by cell
    np.testing.assert_allclose(combine("*", 2.0, a).at_index(0)[1], [2.0, 4.0])
    # the same cells on points of their own (e.g. another run): the same grid
    copy = on_cells([0.0, 1.0], [[1.0, 1.0], [2.0, 2.0]],
                    Mesh(list(SQUARES), [[0, 1, 4, 3], [1, 2, 5, 4]]))
    np.testing.assert_allclose(combine("/", a, copy).at_index(1)[1], [1.5, 2.0])
    # the closest slice of the second, within its time range
    late = on_cells([1.0], [[5.0, 5.0]])
    assert np.isnan(combine("+", a, late).at_index(0)[1]).all()
    np.testing.assert_allclose(combine("+", a, late).at_index(1)[1], [8.0, 9.0])
    # a static grid at the times of a trace
    ramp = Trace(parse_signal_path(TE), np.array([0.0, 1.0, 2.0]),
                 np.array([1.0, 2.0, 3.0]), Axis("ip", "A"))
    scaled = combine("*", on_cells(None, [1.0, 2.0]), ramp)
    assert scaled.count == 3 and scaled.value_axis.units == "eV.A"
    np.testing.assert_allclose(scaled.at_index(2)[1], [3.0, 6.0])


def test_cells_of_other_grids():
    a = on_cells([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])
    three = Mesh(SQUARES + [(1.5, 2.0)], [[0, 1, 4, 3], [1, 2, 5, 4], [4, 5, 6]])
    with pytest.raises(PathError, match=r"different grids \(2 and 3 cells\)"):
        combine("-", a, on_cells([0.0], [[1.0, 2.0, 3.0]], three))
    # the same grid first, and then one that moved: no values then
    moved = Mesh([(r + 1.0, z) for r, z in SQUARES], [[0, 1, 4, 3], [1, 2, 5, 4]])
    b = on_cells([0.0, 1.0], [[1.0, 1.0], [1.0, 1.0]], [TWO_CELLS, moved])
    difference = combine("-", a, b)
    np.testing.assert_allclose(difference.at_index(0)[1], [0.0, 1.0])
    assert np.isnan(difference.at_index(1)[1]).all()
    # other maps and profiles, and derivatives: refused
    x = np.linspace(0.0, 1.0, 3)
    plane = MapSeries(parse_signal_path(TE), None, 1, lambda i: (x, x, np.zeros((3, 3))),
                      Axis("R", "m"), Axis("Z", "m"), Axis("psi", "Wb"))
    with pytest.raises(PathError, match="values on the cells of a grid combine with"):
        combine("-", plane, a)
    with pytest.raises(PathError, match="deriv"):
        derivative(a)


def test_expressions_on_the_demo(source_a, data_dir):
    from idscope.data import DataSource, parse_source

    mesh, te, _ = get(source_a, TE).at(6.0)
    _, ne, _ = get(source_a, NE).at(6.0)  # on the nodes, given to the cells
    per_ev = source_a.signal(parse_signal(f"{NE} / {TE}"))
    assert isinstance(per_ev, MeshSeries) and per_ev.value_axis.units == "m^-3/eV"
    same, values, _ = per_ev.at(6.0)
    assert same is mesh
    np.testing.assert_allclose(values, ne / te)
    # a time trace, at the time of each slice
    ip = get(source_a, "summary/global_quantities/ip/value")
    per_ampere = source_a.signal(parse_signal(f"abs({TE}) / {ip.path.text}"))
    np.testing.assert_allclose(per_ampere.at(6.0)[1],
                               te / np.interp(6.0, ip.time, ip.values))
    # B − A: the demo B is 80 % of A, on the same grid
    source_b = DataSource(parse_source(f"B={data_dir / 'b.nc'}"))
    difference = relative_to(get(source_b, TE), get(source_a, TE), "difference")
    assert difference.value_axis.label.startswith("Δ ")
    np.testing.assert_allclose(difference.at(6.0)[1], -0.2 * te, rtol=1e-9)
    source_b.close()
