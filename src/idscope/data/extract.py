"""Extraction of plottable arrays from an IDS.

A signal path is resolved against the Data Dictionary metadata of the IDS to find
out what it represents:

========================  =========================================================
kind                      examples
========================  =========================================================
``trace``  (y vs time)    ``equilibrium/time_slice/global_quantities/ip``,
                          ``summary/global_quantities/ip/value``,
                          ``magnetics/flux_loop[0]/flux/data``
``profile`` (y vs x)      ``core_profiles/profiles_1d/electrons/temperature``
``map``    (z vs x1, x2)  ``equilibrium/time_slice/profiles_2d[0]/psi``
``shape``  (in R, Z)      ``pf_active/coil[:]/element[:]/geometry``,
                          ``wall/description_2d[0]/limiter/unit[:]/outline``,
                          ``equilibrium/time_slice/boundary/x_point[:]``
========================  =========================================================

Profiles and maps are usually time dependent: they have one curve or image per time
slice and are shown at the time selected in the GUI. Coordinates (time, radial grid,
...) come from the Data Dictionary through ``node.coordinates``, which takes care of
homogeneous and heterogeneous time and of alternative coordinates.

A path with wildcards (``pf_active/coil[:]/current/data``) gives :class:`Channels`:
one trace or profile per element of the arrays of structures, named after them.
"""

import difflib
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from imas.ids_data_type import IDSDataType
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_primitive import IDSPrimitive

from idscope.data.paths import ALL, Index, Range, Selector, SignalPath
from idscope.errors import NoDataError, PathError

logger = logging.getLogger(__name__)

TRACE = "trace"
PROFILE = "profile"
MAP = "map"
SHAPE = "shape"

GEOMETRY, OUTLINE, POINT = "geometry", "outline", "point"
"""Kinds of shapes: closed cross-sections, curves and points."""

_GENERIC_NAMES = {"data", "value", "values", "dim1", "dim2", "dim3"}


class Axis(NamedTuple):
    label: str = ""
    units: str = ""

    def title(self) -> str:
        """Axis title like ``temperature [eV]`` (no units for dimensionless data)."""
        if self.units and self.units not in ("-", "1"):
            return f"{self.label} [{self.units}]"
        return self.label


TIME_AXIS = Axis("time", "s")


class SignalData:
    """Base class of extracted signals.

    Attributes:
        path: the signal path.
        time: time of every slice, or ``None`` if the signal does not depend on time.
        value_axis: label and units of the quantity (y for traces and profiles, z for
            maps).
        documentation: Data Dictionary documentation of the quantity.
        fixed_time: time of an explicitly indexed time slice (e.g. ``profiles_1d[3]``).
    """

    kind = ""

    def __init__(
        self,
        path: SignalPath,
        time: Optional[np.ndarray],
        value_axis: Axis,
        documentation: str = "",
        fixed_time: Optional[float] = None,
    ):
        self.path = path
        self.time = None if time is None else np.asarray(time, dtype=float)
        self.value_axis = value_axis
        self.documentation = documentation
        self.fixed_time = fixed_time

    @property
    def is_time_dependent(self) -> bool:
        return self.time is not None

    def time_index(self, t: Optional[float]) -> Optional[int]:
        """Index of the slice closest to time ``t``.

        Returns 0 for signals that do not depend on time, and ``None`` if there is no
        slice with a finite time.
        """
        if self.time is None:
            return 0
        if len(self.time) == 0:
            return None
        if t is None:
            return 0
        distance = np.abs(self.time - t)
        if np.all(np.isnan(distance)):
            return None
        return int(np.nanargmin(distance))

    def slice_time(self, index: Optional[int]) -> Optional[float]:
        """Time of slice ``index`` (the fixed time for non time-dependent signals)."""
        if self.time is None or index is None:
            return self.fixed_time
        return float(self.time[index])

    def __repr__(self) -> str:
        n = "static" if self.time is None else f"{len(self.time)} times"
        return f"<{type(self).__name__} {self.path.text!r} ({n})>"


class Trace(SignalData):
    """A quantity as a function of time. ``values`` has the same length as ``time``.

    A quantity that does not depend on time has ``time`` set to ``None`` and a
    single value.
    """

    kind = TRACE

    def __init__(self, path, time, values, value_axis, documentation="", fixed_time=None):
        super().__init__(path, time, value_axis, documentation, fixed_time)
        self.values = np.asarray(values, dtype=float)
        self.x_axis = TIME_AXIS


class ProfileSeries(SignalData):
    """A 1D quantity y(x), with one profile per time slice.

    ``xs[i]`` and ``ys[i]`` hold the profile of slice ``i``; the grid may differ from
    slice to slice. Static profiles have ``time`` set to ``None`` and one profile.
    """

    kind = PROFILE

    def __init__(
        self,
        path,
        time,
        xs: Sequence[np.ndarray],
        ys: Sequence[np.ndarray],
        x_axis: Axis,
        value_axis: Axis,
        documentation="",
        fixed_time=None,
    ):
        super().__init__(path, time, value_axis, documentation, fixed_time)
        self.xs = [np.asarray(x, dtype=float) for x in xs]
        self.ys = [np.asarray(y, dtype=float) for y in ys]
        self.x_axis = x_axis

    def at_index(self, index: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
        if index is None or not self.ys:
            return np.empty(0), np.empty(0)
        return self.xs[index], self.ys[index]

    def at(self, t: Optional[float]) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """The profile closest to time ``t``: ``(x, y, time_of_slice)``."""
        index = self.time_index(t)
        x, y = self.at_index(index)
        return x, y, self.slice_time(index)

    def as_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Profiles on a common grid, for a time-space image.

        Returns ``(time, x, z)`` with ``z.shape == (len(time), len(x))``. Profiles are
        interpolated to the grid of the first slice when the grids differ.
        """
        if self.time is None:
            raise ValueError("A static profile cannot be shown as a time-space image.")
        lengths = [len(x) for x in self.xs]
        grid_index = int(np.argmax(lengths)) if lengths else 0
        grid = self.xs[grid_index] if self.xs else np.empty(0)
        z = np.full((len(self.time), len(grid)), np.nan)
        for i, (x, y) in enumerate(zip(self.xs, self.ys)):
            if len(y) == 0 or len(grid) == 0:
                continue
            if len(x) == len(grid) and np.array_equal(x, grid):
                z[i] = y
            else:
                z[i] = _interp(grid, x, y)
        return self.time, grid, z


class MapSeries(SignalData):
    """A 2D quantity z(x1, x2), with one map per time slice.

    Maps are read on demand (they can be large) and cached.
    ``z.shape == (len(x1), len(x2))``: x1 is the horizontal axis.
    """

    kind = MAP

    def __init__(
        self,
        path,
        time,
        count: int,
        loader: Callable[[int], Tuple[np.ndarray, np.ndarray, np.ndarray]],
        x_axis: Axis,
        x2_axis: Axis,
        value_axis: Axis,
        documentation="",
        fixed_time=None,
    ):
        super().__init__(path, time, value_axis, documentation, fixed_time)
        self.count = count
        self._loader = loader
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.x_axis = x_axis
        self.x2_axis = x2_axis

    def at_index(self, index: Optional[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index is None or self.count == 0:
            return np.empty(0), np.empty(0), np.empty((0, 0))
        if index not in self._cache:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[index] = self._loader(index)
        return self._cache[index]

    def at(self, t: Optional[float]):
        """The map closest to time ``t``: ``(x1, x2, z, time_of_slice)``."""
        index = self.time_index(t)
        x1, x2, z = self.at_index(index)
        return x1, x2, z, self.slice_time(index)


class Channels(SignalData):
    """A signal per element of the wildcard arrays of structures of a path.

    ``pf_active/coil[:]/current/data`` has a trace per coil, and
    ``core_profiles/profiles_1d/ion[:]/temperature`` a profile per ion species.
    The members are all traces or all profiles (``kind`` is theirs); :attr:`names`
    come from the ``name``, ``label`` or ``identifier`` of the elements.
    """

    def __init__(
        self,
        path,
        members: Sequence[SignalData],
        names: Sequence[str],
        value_axis: Axis,
        documentation="",
        axis_label: str = "",
    ):
        super().__init__(path, None, value_axis, documentation)
        self.members = list(members)
        self.names = list(names)
        self.axis_label = axis_label
        """What the members are, e.g. ``coil``: the last wildcard of the path."""
        self.kind = self.members[0].kind
        self.x_axis = self.members[0].x_axis

    @property
    def is_time_dependent(self) -> bool:
        return any(member.is_time_dependent for member in self.members)

    def values_at(self, t: Optional[float]) -> Tuple[np.ndarray, Optional[float]]:
        """The value of every trace at the sample closest to time ``t``, and the
        time of that sample (of the first member that has one)."""
        values = np.full(len(self.members), np.nan)
        times = []
        for k, member in enumerate(self.members):
            index = member.time_index(t)
            if index is None:
                continue
            values[k] = member.values if member.time is None else member.values[index]
            times.append(member.slice_time(index))
        known = [time for time in times if time is not None]
        return values, (known[0] if known else None)

    def __repr__(self) -> str:
        return f"<Channels {self.path.text!r} ({len(self.members)} {self.kind}s)>"


class Shapes(SignalData):
    """Shapes in the poloidal plane, drawn over maps: closed cross-sections
    (``geometry``: coils, passive loops), curves (``outline``: walls, the plasma
    boundary) or points (``point``: X-points, the magnetic axis).

    ``slices[i]`` holds an array of (R, Z) rows per member at time slice ``i``
    (static shapes have one slice); rows of NaN separate the parts of a member.
    """

    kind = SHAPE

    def __init__(
        self,
        path,
        time,
        slices: Sequence[Sequence[np.ndarray]],
        shape: str,
        names: Sequence[str],
        documentation="",
        fixed_time=None,
    ):
        super().__init__(path, time, Axis("Z", "m"), documentation, fixed_time)
        self.slices = [list(outlines) for outlines in slices]
        self.shape = shape
        self.names = list(names)
        self.x_axis = Axis("R", "m")
        self.x2_axis = Axis("Z", "m")

    def at(self, t: Optional[float]) -> Tuple[List[np.ndarray], Optional[float]]:
        """The outline of every member at the slice closest to ``t``, and its time."""
        index = self.time_index(t)
        if index is None or not self.slices:
            return [], None
        return self.slices[index], self.slice_time(index)


# ---------------------------------------------------------------------------------
# Path planning against the Data Dictionary metadata
# ---------------------------------------------------------------------------------


class _Step(NamedTuple):
    name: str
    meta: object
    index: Index
    """Index of an array of structures: one element (int), several (Range), or
    ``None`` for a time array of structures without index."""


class _Plan(NamedTuple):
    steps: Tuple[_Step, ...]
    time_pos: Optional[int]
    """Position of the array of structures that provides the time axis."""
    channels: Tuple[int, ...]
    """Positions of the wildcard arrays of structures (other than time)."""
    leaf_meta: object
    shape: Optional[str]
    """``geometry``, ``outline`` or ``point`` for shapes, ``None`` for data."""
    selectors: Tuple[Selector, ...]
    """Selectors on the dimensions of the data node."""
    axes: Optional[Tuple[int, ...]] = None
    """New order of the dimensions left by the selectors (``transpose``)."""


def is_time_aos(meta) -> bool:
    """True if an array of structures (metadata) is indexed by time."""
    if meta.data_type is not IDSDataType.STRUCT_ARRAY:
        return False
    if meta.coordinates and meta.coordinates[0].is_time_coordinate:
        return True
    return meta.type.is_dynamic


def shape_kind(meta) -> Optional[str]:
    """The kind of shape a structure (or the elements of an array of structures)
    describes: a Data Dictionary 2D geometry, an (r, z) curve or an (r, z) point."""
    if meta.data_type not in (IDSDataType.STRUCTURE, IDSDataType.STRUCT_ARRAY):
        return None
    children = {child.name: child for child in meta}
    if {"geometry_type", "outline", "rectangle"} <= set(children):
        return GEOMETRY
    r, z = children.get("r"), children.get("z")
    if r is None or z is None:
        return None
    if r.data_type is not IDSDataType.FLT or z.data_type is not IDSDataType.FLT:
        return None
    if r.ndim == z.ndim == 1:
        return OUTLINE
    if r.ndim == z.ndim == 0:
        return POINT
    return None


def _child_meta(meta, name: str, path_so_far: str):
    try:
        return meta[name]
    except KeyError:
        names = [child.name for child in meta]
        where = f"'{path_so_far}'" if path_so_far else "the IDS"
        msg = f"'{name}' does not exist in {where}"
        close = difflib.get_close_matches(name, names, n=3)
        if close:
            msg += f"; did you mean {', '.join(repr(c) for c in close)}?"
        raise PathError(msg + ".") from None


def plan_path(ids_meta, path: SignalPath) -> _Plan:
    """Check a path against the metadata: find the time axis, the wildcards and
    what the path points to.

    Raises:
        PathError: when the path does not point to plottable data.
    """
    metas = []
    meta = ids_meta
    for pos, name in enumerate(path.parts):
        if pos and meta.data_type not in (IDSDataType.STRUCTURE, IDSDataType.STRUCT_ARRAY):
            here = "/".join(path.parts[:pos])
            raise PathError(f"'{here}' is a data node and has no children.")
        meta = _child_meta(meta, name, "/".join(path.parts[:pos]))
        metas.append(meta)
    leaf = metas[-1]
    here = path.dd_path
    indices = list(path.indices)
    data_index = list(path.data_index)

    shape = None
    if leaf.data_type in (IDSDataType.STRUCTURE, IDSDataType.STRUCT_ARRAY):
        shape = shape_kind(leaf)
        if shape is None and leaf.data_type is IDSDataType.STRUCT_ARRAY:
            raise PathError(
                f"'{here}' is an array of structures; continue the path to a "
                "quantity inside it."
            )
        if shape is None:
            children = ", ".join(sorted(child.name for child in leaf)[:8])
            raise PathError(
                f"'{here}' is a structure; continue the path to one of its "
                f"quantities (e.g. {children})."
            )
        if leaf.data_type is IDSDataType.STRUCT_ARRAY:
            # e.g. boundary/x_point[:]: the index is the one of the array
            indices[-1] = data_index.pop(0) if data_index else None
    elif leaf.data_type not in (IDSDataType.FLT, IDSDataType.INT):
        raise PathError(
            f"'{here}' holds {leaf.data_type.name.lower()} data; only numbers can be "
            "plotted."
        )

    for pos, (meta, index) in enumerate(zip(metas, indices)):
        if index is not None and meta.data_type is not IDSDataType.STRUCT_ARRAY:
            node = "/".join(path.parts[:pos + 1])
            raise PathError(f"'{node}' is a structure and cannot be indexed.")
    # An index applies to the node it is written on: the one of the data node
    # selects along the dimensions of the data, whatever the wildcards before it
    selectors = tuple(data_index)
    if shape is not None and selectors:
        if leaf.data_type is IDSDataType.STRUCT_ARRAY:
            raise PathError(f"'{here}' is an array of structures and takes one index.")
        raise PathError(f"'{here}' is a structure and cannot be indexed.")
    if shape is None and len(selectors) > leaf.ndim:
        if leaf.ndim == 0:
            raise PathError(f"'{here}' is a scalar and cannot be indexed.")
        raise PathError(
            f"'{here}' has {leaf.ndim} dimension{'s' if leaf.ndim != 1 else ''}, but "
            f"{len(selectors)} indices are given."
        )

    time_pos = None
    channels = []
    for pos, (meta, index) in enumerate(zip(metas, indices)):
        if meta.data_type is not IDSDataType.STRUCT_ARRAY or isinstance(index, int):
            continue  # not an array of structures, or one element of it
        node = "/".join(path.parts[:pos + 1])
        name = path.parts[pos]
        if is_time_aos(meta):
            if time_pos is not None:
                raise PathError(
                    f"Only one time-dependent array of structures can be left without "
                    f"an index: '{path.parts[time_pos]}' and '{name}'."
                )
            time_pos = pos
        elif index is None:
            raise PathError(
                f"'{node}' is an array of structures that does not depend on time: "
                f"give an index, e.g. '{name}[0]', or '{name}[:]' for all of them."
            )
        else:
            channels.append(pos)
    steps = tuple(_Step(*item) for item in zip(path.parts, metas, indices))
    axes = _transpose_axes(path, leaf, shape, selectors)
    return _Plan(steps, time_pos, tuple(channels), leaf, shape, selectors, axes)


def _transpose_axes(path: SignalPath, leaf, shape, selectors) -> Optional[Tuple[int, ...]]:
    """The new order of the dimensions left by the index of the data node, for
    ``transpose`` (numpy style: reversed when no axes are given)."""
    if path.transform != "transpose":
        return None
    if shape is not None:
        raise PathError(f"transpose() reorders the dimensions of data; '{path.text}' "
                        "is a shape.")
    full = list(selectors) + [ALL] * (leaf.ndim - len(selectors))
    names = dimension_names(leaf)
    left = [names[d] for d, selector in enumerate(full) if isinstance(selector, Range)]
    if len(left) < 2:
        raise PathError(
            f"'{path.text}': '{path.dd_path}' has {len(left)} dimension"
            f"{'s' if len(left) != 1 else ''} left, nothing to transpose."
        )
    axes = path.transform_args or tuple(reversed(range(len(left))))
    if sorted(axes) != list(range(len(left))):
        raise PathError(
            f"'{path.text}': transpose() needs a new order of the {len(left)} "
            f"dimensions ({', '.join(left)}), i.e. the numbers 0 to {len(left) - 1} "
            f"each once, not {', '.join(map(str, axes))}."
        )
    return tuple(axes)


# ---------------------------------------------------------------------------------
# Reading data
# ---------------------------------------------------------------------------------


def _as_float_array(value, data_type) -> np.ndarray:
    array = np.array(value, dtype=float)
    empty = EMPTY_INT if data_type is IDSDataType.INT else EMPTY_FLOAT
    if array.ndim == 0:
        return np.array(np.nan) if array == empty else array
    array[array == empty] = np.nan
    return array


def _leaf_array(leaf) -> np.ndarray:
    """Value of a data node as a float array (empty values become NaN)."""
    meta = leaf.metadata
    return _as_float_array(leaf.value, meta.data_type)


def _elements(aos, index: Index, what: str) -> Union[int, range]:
    """The element (int) or elements (range) of an array of structures that
    ``index`` picks (all of them for ``None``)."""
    selection = range(len(aos))
    if index is None:
        return selection
    if isinstance(index, Range):
        return selection[index.as_slice()]
    size = len(selection)
    if not -size <= index < size:
        if size == 0:
            raise NoDataError(f"'{what}' is empty.")
        raise NoDataError(
            f"Index {index} is out of range for '{what}', which has {size} "
            f"element{'s' if size != 1 else ''}."
        )
    return selection[index]


def _walk(node, plan: _Plan, start: int, stop: int, key: Dict[int, int],
          times_out: List[float]):
    """Follow the steps ``start`` to ``stop`` of a plan from ``node``.

    Wildcard arrays of structures take their index from ``key``, the others pick
    one element. The time of explicitly indexed time slices is appended to
    ``times_out``.
    """
    for pos in range(start, stop):
        step = plan.steps[pos]
        child = getattr(node, step.name)
        if step.meta.data_type is IDSDataType.STRUCT_ARRAY:
            what = child.metadata.path_string
            if pos in key:
                index = key[pos]
                if index >= len(child):
                    raise NoDataError(f"'{what}' has no element {index}.")
            else:
                index = _elements(child, step.index, what)
                if is_time_aos(child.metadata):
                    times_out.append(float(aos_times(child)[index]))
            child = child[index]
        node = child
    return node


def aos_times(aos) -> np.ndarray:
    """Time of every element of a time-dependent array of structures."""
    n = len(aos)
    times = None
    try:
        coordinate = aos.coordinates[0]
        values = coordinate.value if isinstance(coordinate, IDSPrimitive) else coordinate
        times = np.array(values, dtype=float)
    except Exception:
        logger.debug("No time coordinate for %s", aos.metadata.path_string, exc_info=True)
    if times is None or times.shape != (n,):
        try:
            times = np.array([float(element.time) for element in aos], dtype=float)
        except Exception:
            times = None
    if times is None or times.shape != (n,):
        logger.warning(
            "No valid time for '%s'; using the slice index instead.",
            aos.metadata.path_string,
        )
        times = np.arange(n, dtype=float)
    times[times == EMPTY_FLOAT] = np.nan
    return times


def _element_name(element, aos_name: str, index: int) -> str:
    """Name of an element of an array of structures, for legends."""
    for child in ("name", "label", "identifier"):
        node = getattr(element, child, None)
        if node is None:
            continue
        if not isinstance(node, IDSPrimitive):  # e.g. an identifier structure
            node = getattr(node, "name", None)
            if not isinstance(node, IDSPrimitive):
                continue
        value = node.value
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{aos_name}[{index}]"


def _members(ids, plan: _Plan) -> List[Tuple[Dict[int, int], str]]:
    """The index in every wildcard array of structures, and the name, of each
    member of a path with wildcards."""
    found: Dict[Tuple[int, ...], Tuple[Tuple[str, int], ...]] = {}
    names: Dict[Tuple[int, ...], Tuple[str, int]] = {}
    deepest = plan.channels[-1]

    def walk(node, pos, key, labels):
        if pos > deepest:
            found.setdefault(key, labels)
            return
        step = plan.steps[pos]
        child = getattr(node, step.name)
        if step.meta.data_type is not IDSDataType.STRUCT_ARRAY:
            walk(child, pos + 1, key, labels)
            return
        what = child.metadata.path_string
        if pos in plan.channels:
            selection = _elements(child, step.index, what)
            for i in selection:
                sub = key + (i,)
                if sub not in names:
                    names[sub] = (_element_name(child[i], step.name, i), len(selection))
                walk(child[i], pos + 1, sub, labels + (names[sub],))
        elif pos == plan.time_pos:
            for i in _elements(child, step.index, what):
                walk(child[i], pos + 1, key, labels)
        else:
            try:
                index = _elements(child, step.index, what)
            except NoDataError:
                return
            walk(child[index], pos + 1, key, labels)

    walk(ids, 0, (), ())
    members = []
    seen: Dict[str, int] = {}
    for key in sorted(found):
        labels = found[key]
        # Levels with a single element (e.g. coils of one element) add nothing
        shown = [name for name, count in labels if count > 1] or [labels[-1][0]]
        name = "/".join(shown)
        # Elements can share a name, e.g. two core_sources both "nbi": nbi, nbi#2
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name}#{seen[name]}"
        members.append((dict(zip(plan.channels, key)), name))
    return members


def coordinate_label(meta) -> str:
    parts = meta.path_string.split("/")
    if meta.name in _GENERIC_NAMES and len(parts) > 1:
        return "/".join(parts[-2:])
    return meta.name


def _coordinate(leaf, dim: int, size: int) -> Tuple[np.ndarray, Axis]:
    """Coordinate values and axis of dimension ``dim`` of a data node."""
    try:
        coordinate = leaf.coordinates[dim]
    except Exception:
        logger.debug(
            "Cannot resolve coordinate %d of %s", dim, leaf.metadata.path_string,
            exc_info=True,
        )
        coordinate = None
    if isinstance(coordinate, IDSPrimitive):
        values = _as_float_array(coordinate.value, coordinate.metadata.data_type)
        if values.shape == (size,):
            axis = Axis(coordinate_label(coordinate.metadata), coordinate.metadata.units)
            return values, axis
    return np.arange(size, dtype=float), Axis("index", "")


def _time_of(leaf, dim: int, size: int) -> np.ndarray:
    """Time values of dimension ``dim`` of a data node."""
    values, axis = _coordinate(leaf, dim, size)
    if axis.label == "index":
        logger.warning(
            "No valid time for '%s'; using the sample index instead.",
            leaf.metadata.path_string,
        )
    return values


def _full_selectors(selectors: Sequence[Selector], shape, what: str) -> list:
    """One selector per dimension of an array of ``shape`` (indices made positive)."""
    full = list(selectors) + [ALL] * (len(shape) - len(selectors))
    for dim, selector in enumerate(full):
        if isinstance(selector, Range):
            continue
        size = shape[dim]
        if not -size <= selector < size:
            raise NoDataError(f"Index {selector} is out of range for '{what}' (size {size}).")
        full[dim] = selector % size
    return full


def _numpy_index(full) -> tuple:
    return tuple(s.as_slice() if isinstance(s, Range) else s for s in full)


def extract_signal(ids, path: SignalPath) -> SignalData:
    """Extract a signal from an IDS (typically lazily loaded).

    Raises:
        PathError: when the path does not match the Data Dictionary.
        NoDataError: when the data is not filled.
    """
    data = _extract(ids, path)  # transpose() is applied while reading
    if path.transform == "norm":
        data = normalize(data)
    return data


def normalize(data: SignalData) -> SignalData:
    """``(y - y[0]) / (y[-1] - y[0])`` for every profile, e.g. psi -> psi_n."""
    if isinstance(data, Channels) and data.kind == PROFILE:
        members, names = [], []
        for member, name in zip(data.members, data.names):
            try:
                members.append(normalize(member))
                names.append(name)
            except NoDataError:
                continue
        if not members:
            raise NoDataError(
                f"'{data.path.text}' cannot be normalized: the first and last values "
                "of its profiles are missing or equal."
            )
        return Channels(data.path, members, names, members[0].value_axis,
                        data.documentation, data.axis_label)
    if not isinstance(data, ProfileSeries):
        raise PathError(
            f"norm() applies to profiles, but '{data.path.text}' is a {data.kind}."
        )
    ys = [_normalized(y) for y in data.ys]
    if not any(np.isfinite(y).any() for y in ys):
        raise NoDataError(
            f"'{data.path.text}' cannot be normalized: its first and last values are "
            "missing or equal."
        )
    return ProfileSeries(
        data.path, data.time, data.xs, ys, data.x_axis, Axis(data.path.name, ""),
        data.documentation, data.fixed_time,
    )


def _normalized(y: np.ndarray) -> np.ndarray:
    if len(y) < 2 or not (np.isfinite(y[0]) and np.isfinite(y[-1])) or y[0] == y[-1]:
        return np.full(len(y), np.nan)
    return (y - y[0]) / (y[-1] - y[0])


def _extract(ids, path: SignalPath) -> SignalData:
    plan = plan_path(ids.metadata, path)
    if not plan.channels:
        return _extract_member(ids, path, plan, {})
    members = _members(ids, plan)
    if not members:
        first = "/".join(path.parts[:plan.channels[0] + 1])
        raise NoDataError(f"'{path.text}': '{first}' has no elements.")
    data, names, errors = [], [], []
    for key, name in members:
        try:
            data.append(_extract_member(ids, path, plan, key))
            names.append(name)
        except NoDataError as exc:
            errors.append(exc)
    if not data:
        raise errors[0]
    if plan.shape is not None:
        slices = [[member.slices[i][0] for member in data]
                  for i in range(len(data[0].slices))]
        return Shapes(path, data[0].time, slices, plan.shape, names,
                      data[0].documentation, data[0].fixed_time)
    return Channels(path, data, names, data[0].value_axis, data[0].documentation,
                    path.parts[plan.channels[-1]])


def _extract_member(ids, path: SignalPath, plan: _Plan, key: Dict[int, int]):
    """The signal of one member of a path (the whole signal without wildcards)."""
    meta = plan.leaf_meta
    doc = meta.documentation or ""
    value_axis = Axis(path.label, meta.units if plan.shape is None else "m")
    fixed_times: List[float] = []
    n_steps = len(plan.steps)

    if plan.time_pos is None:
        node = _walk(ids, plan, 0, n_steps, key, fixed_times)
        fixed_time = fixed_times[-1] if fixed_times else None
        if plan.shape is not None:
            outline = read_shape(node, plan.shape)
            if outline is None:
                raise NoDataError(f"'{path.text}' is not filled.")
            return Shapes(path, None, [[outline]], plan.shape, [""], doc, fixed_time)
        return _from_leaf(path, node, plan.selectors, value_axis, doc, fixed_time,
                          plan.axes)

    tp = plan.time_pos
    parent = _walk(ids, plan, 0, tp, key, fixed_times)
    fixed_time = fixed_times[-1] if fixed_times else None
    aos = getattr(parent, plan.steps[tp].name)
    slices = list(_elements(aos, plan.steps[tp].index, aos.metadata.path_string))
    if not slices:
        raise NoDataError(f"'{aos.metadata.path_string}' has no time slices.")
    times = aos_times(aos)[slices]
    errors: List[NoDataError] = []

    def node_of(i: int):
        """The node the path points to in time slice i (None if it is missing)."""
        try:
            return _walk(aos[i], plan, tp + 1, n_steps, key, [])
        except NoDataError as exc:  # e.g. fewer ions in this slice
            errors.append(exc)
            return None

    def not_filled() -> NoDataError:
        if errors:
            return NoDataError(f"'{path.text}': {errors[0]}")
        return NoDataError(f"'{path.text}' is not filled.")

    if plan.shape is not None:
        outlines = []
        for i in slices:
            node = node_of(i)
            outline = read_shape(node, plan.shape) if node is not None else None
            outlines.append([np.empty((0, 2)) if outline is None else outline])
        if all(outline[0].size == 0 for outline in outlines):
            raise not_filled()
        return Shapes(path, times, outlines, plan.shape, [""], doc, fixed_time)

    full = list(plan.selectors) + [ALL] * (meta.ndim - len(plan.selectors))
    kept = [dim for dim, selector in enumerate(full) if isinstance(selector, Range)]
    if plan.axes is not None:  # transpose(): the dimensions in their new order
        kept = [kept[axis] for axis in plan.axes]

    def slice_data(i: int):
        """Data node, value (after the selectors) and selectors of slice i, or
        Nones when it is not filled."""
        leaf = node_of(i)
        if leaf is None:
            return None, None, None
        value = _leaf_array(leaf)
        if value.size == 0 or (value.ndim == 0 and np.isnan(value)):
            return None, None, None
        try:
            selectors = _full_selectors(plan.selectors, value.shape, meta.path_string)
        except NoDataError as exc:
            errors.append(exc)
            return None, None, None
        selected = value[_numpy_index(selectors)]
        if plan.axes is not None:
            selected = np.transpose(selected, plan.axes)
        return leaf, selected, selectors

    def coordinate(leaf, dim: int, selectors, size: int):
        x, axis = _coordinate(leaf, dim, size)
        return x[selectors[dim].as_slice()], axis

    if not kept:
        values = np.full(len(slices), np.nan)
        for k, i in enumerate(slices):
            _, value, _ = slice_data(i)
            if value is not None:
                values[k] = value
        if np.all(np.isnan(values)):
            raise not_filled()
        return Trace(path, times, values, value_axis, doc, fixed_time)

    if len(kept) == 1:
        xs, ys, x_axis = [], [], None
        for i in slices:
            leaf, y, selectors = slice_data(i)
            if leaf is None:
                xs.append(np.empty(0))
                ys.append(np.empty(0))
                continue
            size = _leaf_array(leaf).shape[kept[0]]
            x, axis = coordinate(leaf, kept[0], selectors, size)
            xs.append(x)
            ys.append(y)
            x_axis = x_axis or axis
        if x_axis is None:
            raise not_filled()
        return ProfileSeries(path, times, xs, ys, x_axis, value_axis, doc, fixed_time)

    if len(kept) == 2:
        axes = _map_axes(meta, kept)

        def load(k: int):
            leaf, z, selectors = slice_data(slices[k])
            if leaf is None:
                return np.empty(0), np.empty(0), np.empty((0, 0))
            shape = _leaf_array(leaf).shape
            x1, _ = coordinate(leaf, kept[0], selectors, shape[kept[0]])
            x2, _ = coordinate(leaf, kept[1], selectors, shape[kept[1]])
            return x1, x2, z

        return MapSeries(path, times, len(slices), load, axes[0], axes[1], value_axis,
                         doc, fixed_time)

    raise PathError(
        f"'{path.text}' has {len(kept)} dimensions per time slice; idscope can show up "
        "to 2."
    )


def dimension_names(meta) -> List[str]:
    """What the dimensions of a data node run along, in order: the names of its
    Data Dictionary coordinates, e.g. ``['grid/dim1', 'grid/dim2']`` for a psi map.
    ``time`` for a time coordinate, ``index`` for a plain 1...N."""
    names = []
    for coordinate in meta.coordinates:
        if coordinate.is_time_coordinate:
            names.append("time")
        elif coordinate.references:
            parts = coordinate.references[0].parts
            names.append("/".join(parts[-2:]) if parts[-1] in _GENERIC_NAMES else parts[-1])
        else:
            names.append("index")
    return names


def _map_axes(meta, dims) -> Tuple[Axis, Axis]:
    """Axis labels of a map from the metadata of its coordinates."""
    names = dimension_names(meta)
    return Axis(names[dims[0]], ""), Axis(names[dims[1]], "")


def _from_leaf(path, leaf, selectors, value_axis, doc, fixed_time,
               axes: Optional[Tuple[int, ...]] = None) -> SignalData:
    """Build a signal from a single data node (no time array of structures).

    The dimensions left by the selectors keep their order, or take the one of
    ``axes`` (transpose), and none has a role of its own, time included: one
    dimension is a curve along its coordinate (a time trace when that is time), two
    are a map, the first along x.
    """
    meta = leaf.metadata
    value = _leaf_array(leaf)
    if value.size == 0 or (value.ndim == 0 and np.isnan(value)):
        raise NoDataError(f"'{path.text}' is not filled.")
    full = _full_selectors(selectors, value.shape, meta.path_string)
    for dim in range(value.ndim):
        if meta.coordinates[dim].is_time_coordinate and not isinstance(full[dim], Range):
            # One time sample: its time, for titles (the data is not changed)
            fixed_time = float(_time_of(leaf, dim, value.shape[dim])[full[dim]])
    kept = [d for d, selector in enumerate(full) if isinstance(selector, Range)]
    selected = value[_numpy_index(full)]
    if axes is not None:
        selected = np.transpose(selected, axes)
        kept = [kept[axis] for axis in axes]

    def coordinate(dim: int):
        x, axis = _coordinate(leaf, dim, value.shape[dim])
        return x[full[dim].as_slice()], axis

    if len(kept) > 2:
        names = dimension_names(meta)
        raise PathError(
            f"'{path.text}' has {len(kept)} dimensions "
            f"({', '.join(names[d] for d in kept)}); idscope shows up to 2: select "
            "some with an index on the data, e.g. [:, :, 0]."
        )
    if not kept:
        return Trace(path, None, selected, value_axis, doc, fixed_time)
    if len(kept) == 1:
        (dim,) = kept
        if meta.coordinates[dim].is_time_coordinate:  # a curve along time
            time = _time_of(leaf, dim, value.shape[dim])[full[dim].as_slice()]
            return Trace(path, time, selected, value_axis, doc, fixed_time)
        x, x_axis = coordinate(dim)
        return ProfileSeries(path, None, [x], [selected], x_axis, value_axis, doc,
                             fixed_time)
    (x1, a1), (x2, a2) = coordinate(kept[0]), coordinate(kept[1])
    return MapSeries(path, None, 1, lambda i: (x1, x2, selected), a1, a2,
                     value_axis, doc, fixed_time)


# ---------------------------------------------------------------------------------
# Shapes in the poloidal plane
# ---------------------------------------------------------------------------------

_CIRCLE_POINTS = 65


def read_shape(node, shape: str) -> Optional[np.ndarray]:
    """(R, Z) rows of a shape structure, or None when it is not filled."""
    if shape == GEOMETRY:
        return geometry_outline(node)
    r, z = _leaf_array(node.r), _leaf_array(node.z)
    if shape == POINT:
        if not (np.isfinite(r) and np.isfinite(z)):
            return None
        return np.array([[float(r), float(z)]])
    if r.size == 0 or r.shape != z.shape:
        return None
    points = np.column_stack([r, z])
    return _closed(points) if _is_closed(node, points) else points


def _is_closed(outline, points: np.ndarray) -> bool:
    """Whether a curve is closed: as its ``closed`` flag says (e.g. limiter units
    of DD 3), else when its ends are about one segment apart."""
    try:
        from imas.util import get_parent

        flag = getattr(get_parent(outline), "closed", None)
    except Exception:  # pragma: no cover - depends on the imas-python version
        flag = None
    if isinstance(flag, IDSPrimitive) and flag.has_value:
        return int(flag.value) == 1
    finite = points[np.all(np.isfinite(points), axis=1)]
    if len(finite) < 3:
        return False
    steps = np.hypot(*np.diff(finite, axis=0).T)
    gap = float(np.hypot(*(finite[-1] - finite[0])))
    return gap <= 2.0 * float(np.median(steps))


def _scalar(node) -> float:
    value = _leaf_array(node)
    return float(value) if value.ndim == 0 else float("nan")


def _closed(points) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) and not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[:1]])
    return points


def _circle(r: float, z: float, radius: float) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, _CIRCLE_POINTS)
    return np.column_stack([r + radius * np.cos(theta), z + radius * np.sin(theta)])


def _arc(start, end, radius: float) -> np.ndarray:
    """Points from ``start`` towards ``end`` (excluded) along an arc of circle of
    the given radius (a straight line for 0).

    The Data Dictionary does not define the sign of the curvature radius: a positive
    radius is taken to bend the arc to the right of the direction of travel (outwards
    for a counter-clockwise contour), a negative one to the left.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    chord = end - start
    length = float(np.hypot(*chord))
    if not np.isfinite(radius) or radius == 0 or length == 0 or abs(radius) < length / 2:
        return start[None, :]
    left = np.array([-chord[1], chord[0]]) / length
    # The center is on the other side of the chord than the (shorter) arc
    center = (start + end) / 2 + np.sign(radius) * left * np.sqrt(radius**2 - (length / 2) ** 2)
    a0 = np.arctan2(*(start - center)[::-1])
    a1 = np.arctan2(*(end - center)[::-1])
    sweep = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi  # the short way round
    angles = a0 + sweep * np.linspace(0.0, 1.0, 17)[:-1]
    return center + abs(radius) * np.column_stack([np.cos(angles), np.sin(angles)])


def geometry_outline(geometry) -> Optional[np.ndarray]:
    """Closed outline of a Data Dictionary 2D geometry (of coil elements, passive
    loops, ...), for any of its ``geometry_type``: outline, rectangle, oblique
    (parallelogram), arcs of circle, annulus or thick line. Rows of NaN separate the
    two circles of an annulus. None when it is not filled."""
    kind = _scalar(geometry.geometry_type)
    kind = int(kind) if np.isfinite(kind) else 0
    outline = geometry.outline
    rectangle = geometry.rectangle
    if kind == 1 or (kind == 0 and len(outline.r)):
        r, z = _leaf_array(outline.r), _leaf_array(outline.z)
        if r.size == 0 or r.shape != z.shape:
            return None
        return _closed(np.column_stack([r, z]))
    if kind in (0, 2):
        r, z = _scalar(rectangle.r), _scalar(rectangle.z)
        width, height = _scalar(rectangle.width), _scalar(rectangle.height)
        if not np.all(np.isfinite([r, z, width, height])):
            return None
        dr, dz = width / 2, height / 2
        return _closed([[r - dr, z - dz], [r + dr, z - dz], [r + dr, z + dz],
                        [r - dr, z + dz]])
    if kind == 3:
        oblique = geometry.oblique
        r, z = _scalar(oblique.r), _scalar(oblique.z)
        a, b = _scalar(oblique.length_alpha), _scalar(oblique.length_beta)
        alpha, beta = (np.nan_to_num(_scalar(oblique.alpha)),
                       np.nan_to_num(_scalar(oblique.beta)))
        if not np.all(np.isfinite([r, z, a, b])):
            return None
        p0 = np.array([r, z])
        p1 = p0 + a * np.array([np.cos(alpha), np.sin(alpha)])
        p3 = p0 + b * np.array([-np.sin(beta), np.cos(beta)])
        return _closed([p0, p1, p1 + p3 - p0, p3])
    if kind == 4:
        arcs = geometry.arcs_of_circle
        r, z = _leaf_array(arcs.r), _leaf_array(arcs.z)
        radii = _leaf_array(arcs.curvature_radii)
        if r.size < 2 or r.shape != z.shape:
            return None
        points = np.column_stack([r, z])
        parts = [
            _arc(points[i], points[(i + 1) % len(points)],
                 float(radii[i]) if i < radii.size else 0.0)
            for i in range(len(points))
        ]
        return _closed(np.vstack(parts))
    if kind == 5:
        annulus = geometry.annulus
        r, z = _scalar(annulus.r), _scalar(annulus.z)
        inner, outer = _scalar(annulus.radius_inner), _scalar(annulus.radius_outer)
        if not np.all(np.isfinite([r, z, outer])):
            return None
        rings = [_circle(r, z, outer)]
        if np.isfinite(inner) and inner > 0:
            rings += [np.full((1, 2), np.nan), _circle(r, z, inner)]
        return np.vstack(rings)
    if kind == 6:
        line = geometry.thick_line
        p1 = np.array([_scalar(line.first_point.r), _scalar(line.first_point.z)])
        p2 = np.array([_scalar(line.second_point.r), _scalar(line.second_point.z)])
        thickness = _scalar(line.thickness)
        length = float(np.hypot(*(p2 - p1)))
        if not np.all(np.isfinite([*p1, *p2, thickness])) or length == 0:
            return None
        normal = np.array([-(p2 - p1)[1], (p2 - p1)[0]]) / length * thickness / 2
        return _closed([p1 + normal, p2 + normal, p2 - normal, p1 - normal])
    return None


def apply_x(data: SignalData, x_data: SignalData) -> SignalData:
    """Use the values of ``x_data`` as x coordinate of the profiles in ``data``.

    Both must be profiles with the same number of slices and points, for example
    ``profiles_1d/electrons/temperature`` with ``profiles_1d/grid/psi``.
    """
    if isinstance(data, Channels):
        if isinstance(x_data, Channels):
            pairs = _paired(data, x_data)
            members = [apply_x(member, x_member) for member, x_member in pairs]
        else:
            members = [apply_x(member, x_data) for member in data.members]
        return Channels(data.path, members, data.names, data.value_axis,
                        data.documentation, data.axis_label)
    if isinstance(x_data, Channels):
        raise PathError(
            f"The x path '{x_data.path.text}' has wildcards, but '{data.path.text}' "
            "does not."
        )
    if not isinstance(data, ProfileSeries) or not isinstance(x_data, ProfileSeries):
        raise PathError("An x path is only supported for profiles.")
    if x_data.time is not None and data.time is not None:
        if len(x_data.ys) != len(data.ys):
            raise PathError(
                f"'{x_data.path.text}' has {len(x_data.ys)} time slices but "
                f"'{data.path.text}' has {len(data.ys)}."
            )
        new_xs = x_data.ys
    elif x_data.time is None:
        new_xs = [x_data.ys[0]] * len(data.ys)
    else:
        raise PathError(
            f"'{x_data.path.text}' depends on time but '{data.path.text}' does not."
        )
    for i, (x, y) in enumerate(zip(new_xs, data.ys)):
        if len(y) and len(x) != len(y):
            raise PathError(
                f"'{x_data.path.text}' and '{data.path.text}' have different lengths "
                f"({len(x)} and {len(y)}) in slice {i}."
            )
    x_axis = Axis(x_data.path.name, x_data.value_axis.units)
    return ProfileSeries(
        data.path, data.time, new_xs, data.ys, x_axis, data.value_axis,
        data.documentation, data.fixed_time,
    )


def _paired(data: Channels, x_data: Channels):
    """Members of ``data`` with the members of ``x_data`` of the same name, or in
    the same position."""
    x_by_name = dict(zip(x_data.names, x_data.members))
    if set(data.names) <= set(x_by_name):
        return [(member, x_by_name[name]) for member, name in zip(data.members, data.names)]
    if len(data.members) != len(x_data.members):
        raise PathError(
            f"'{x_data.path.text}' has {len(x_data.members)} elements but "
            f"'{data.path.text}' has {len(data.members)}."
        )
    return list(zip(data.members, x_data.members))


def _interp(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y(x) on ``grid``, NaN outside of the data range."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order], left=np.nan, right=np.nan)
