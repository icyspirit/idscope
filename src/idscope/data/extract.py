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
========================  =========================================================

Profiles and maps are usually time dependent: they have one curve or image per time
slice and are shown at the time selected in the GUI. Coordinates (time, radial grid,
...) come from the Data Dictionary through ``node.coordinates``, which takes care of
homogeneous and heterogeneous time and of alternative coordinates.
"""

import difflib
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from imas.ids_data_type import IDSDataType
from imas.ids_defs import EMPTY_FLOAT, EMPTY_INT
from imas.ids_primitive import IDSPrimitive

from idscope.data.paths import ALL, SignalPath
from idscope.errors import NoDataError, PathError

logger = logging.getLogger(__name__)

TRACE = "trace"
PROFILE = "profile"
MAP = "map"

_GENERIC_NAMES = {"data", "value", "values", "dim1", "dim2", "dim3"}


class Axis(NamedTuple):
    label: str = ""
    units: str = ""

    def title(self) -> str:
        """Axis title like ``temperature [eV]``."""
        if self.units and self.units != "-":
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


# ---------------------------------------------------------------------------------
# Path planning against the Data Dictionary metadata
# ---------------------------------------------------------------------------------


class _Plan(NamedTuple):
    time_pos: Optional[int]
    """Position in the path of the array of structures that provides the time axis."""
    leaf_meta: object
    leaf_index: Optional[int]


def is_time_aos(meta) -> bool:
    """True if an array of structures (metadata) is indexed by time."""
    if meta.data_type is not IDSDataType.STRUCT_ARRAY:
        return False
    if meta.coordinates and meta.coordinates[0].is_time_coordinate:
        return True
    return meta.type.is_dynamic


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
    """Check a path against the metadata and find the time axis.

    Raises:
        PathError: when the path does not point to plottable data.
    """
    meta = ids_meta
    time_pos = None
    walked: List[str] = []
    last = len(path.parts) - 1
    for pos, (name, index) in enumerate(zip(path.parts, path.indices)):
        meta = _child_meta(meta, name, "/".join(walked))
        walked.append(name)
        here = "/".join(walked)
        data_type = meta.data_type
        if data_type is IDSDataType.STRUCT_ARRAY:
            if pos == last:
                raise PathError(
                    f"'{here}' is an array of structures; continue the path to a "
                    "quantity inside it."
                )
            if index is None or index is ALL:
                if not is_time_aos(meta):
                    raise PathError(
                        f"'{here}' is an array of structures that does not depend on "
                        f"time: give an index, e.g. '{name}[0]'."
                    )
                if time_pos is not None:
                    raise PathError(
                        f"Only one time-dependent array of structures can be left "
                        f"without an index: '{path.parts[time_pos]}' and '{name}'."
                    )
                time_pos = pos
        elif data_type is IDSDataType.STRUCTURE:
            if index is not None:
                raise PathError(f"'{here}' is a structure and cannot be indexed.")
            if pos == last:
                children = ", ".join(sorted(child.name for child in meta)[:8])
                raise PathError(
                    f"'{here}' is a structure; continue the path to one of its "
                    f"quantities (e.g. {children})."
                )
        else:
            if pos != last:
                raise PathError(f"'{here}' is a data node and has no children.")
            if data_type not in (IDSDataType.FLT, IDSDataType.INT):
                raise PathError(
                    f"'{here}' holds {data_type.name.lower()} data; only numbers can "
                    "be plotted."
                )
            if index is ALL:
                index = None
            if index is not None and meta.ndim == 0:
                raise PathError(f"'{here}' is a scalar and cannot be indexed.")
            return _Plan(time_pos, meta, index)
    raise PathError(f"'{path.text}' does not point to a quantity.")  # pragma: no cover


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


def _navigate(node, parts, indices, times_out: List[float]):
    """Follow ``parts``/``indices`` from ``node``, checking array bounds.

    Only arrays of structures are indexed: an index on the data node is applied
    later, by :func:`_select`. The time of explicitly indexed time slices is appended
    to ``times_out``.
    """
    for name, index in zip(parts, indices):
        child = getattr(node, name)
        if isinstance(index, int) and child.metadata.data_type is IDSDataType.STRUCT_ARRAY:
            size = len(child)
            if index >= size:
                what = child.metadata.path_string
                if size == 0:
                    raise NoDataError(f"'{what}' is empty.")
                raise NoDataError(
                    f"Index {index} is out of range for '{what}', which has {size} "
                    f"element{'s' if size != 1 else ''}."
                )
            if is_time_aos(child.metadata):
                times = aos_times(child)
                times_out.append(float(times[index]))
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


def _select(value: np.ndarray, index: Optional[int], what: str) -> np.ndarray:
    """Apply an index on the first dimension of a data node."""
    if index is None:
        return value
    if index >= value.shape[0]:
        raise NoDataError(
            f"Index {index} is out of range for '{what}' (size {value.shape[0]})."
        )
    return value[index]


def extract_signal(ids, path: SignalPath) -> SignalData:
    """Extract a signal from an IDS (typically lazily loaded).

    Raises:
        PathError: when the path does not match the Data Dictionary.
        NoDataError: when the data is not filled.
    """
    plan = plan_path(ids.metadata, path)
    meta = plan.leaf_meta
    value_axis = Axis(path.label, meta.units)
    doc = meta.documentation or ""
    fixed_times: List[float] = []

    if plan.time_pos is None:
        leaf = _navigate(ids, path.parts, path.indices, fixed_times)
        fixed_time = fixed_times[-1] if fixed_times else None
        return _from_leaf(path, leaf, plan, value_axis, doc, fixed_time)

    split = plan.time_pos + 1
    aos = _navigate(ids, path.parts[:split], path.indices[:plan.time_pos] + (None,),
                    fixed_times)
    fixed_time = fixed_times[-1] if fixed_times else None
    if len(aos) == 0:
        raise NoDataError(f"'{aos.metadata.path_string}' has no time slices.")
    times = aos_times(aos)
    parts, indices = path.parts[split:], path.indices[split:]
    errors: List[NoDataError] = []

    def slice_data(i: int):
        """Data node and value of slice i, or (None, None) when it is not filled."""
        try:
            leaf = _navigate(aos[i], parts, indices, [])
            value = _leaf_array(leaf)
            if value.size == 0 or (value.ndim == 0 and np.isnan(value)):
                return None, None
            return leaf, _select(value, plan.leaf_index, meta.path_string)
        except NoDataError as exc:  # e.g. fewer ions in this slice
            errors.append(exc)
            return None, None

    def not_filled() -> NoDataError:
        if errors:
            return NoDataError(f"'{path.text}': {errors[0]}")
        return NoDataError(f"'{path.text}' is not filled.")

    ndim = meta.ndim - (plan.leaf_index is not None)
    if ndim == 0:
        values = np.full(len(aos), np.nan)
        for i in range(len(aos)):
            _, value = slice_data(i)
            if value is not None:
                values[i] = value
        if np.all(np.isnan(values)):
            raise not_filled()
        return Trace(path, times, values, value_axis, doc, fixed_time)

    if ndim == 1:
        dim = 1 if plan.leaf_index is not None else 0
        xs, ys, x_axis = [], [], None
        for i in range(len(aos)):
            leaf, y = slice_data(i)
            if leaf is None:
                xs.append(np.empty(0))
                ys.append(np.empty(0))
                continue
            x, axis = _coordinate(leaf, dim, len(y))
            xs.append(x)
            ys.append(y)
            x_axis = x_axis or axis
        if x_axis is None:
            raise not_filled()
        return ProfileSeries(path, times, xs, ys, x_axis, value_axis, doc, fixed_time)

    if ndim == 2:
        dims = (1, 2) if plan.leaf_index is not None else (0, 1)
        axes = _map_axes(meta, dims)

        def load(i: int):
            leaf, z = slice_data(i)
            if leaf is None:
                return np.empty(0), np.empty(0), np.empty((0, 0))
            x1, _ = _coordinate(leaf, dims[0], z.shape[0])
            x2, _ = _coordinate(leaf, dims[1], z.shape[1])
            return x1, x2, z

        return MapSeries(path, times, len(aos), load, axes[0], axes[1], value_axis,
                         doc, fixed_time)

    raise PathError(
        f"'{path.text}' has {ndim} dimensions per time slice; idscope can show up "
        "to 2."
    )


def _map_axes(meta, dims) -> Tuple[Axis, Axis]:
    """Axis labels of a map from the metadata of its coordinates."""
    axes = []
    for dim in dims:
        coordinate = meta.coordinates[dim]
        if coordinate.references:
            ref = coordinate.references[0]
            name = ref.parts[-1]
            label = "/".join(ref.parts[-2:]) if name in _GENERIC_NAMES else name
            axes.append(Axis(label, ""))
        else:
            axes.append(Axis("index", ""))
    return axes[0], axes[1]


def _from_leaf(path, leaf, plan: _Plan, value_axis, doc, fixed_time) -> SignalData:
    """Build a signal from a single data node (no time array of structures)."""
    meta = leaf.metadata
    value = _leaf_array(leaf)
    if value.size == 0 or (value.ndim == 0 and np.isnan(value)):
        raise NoDataError(f"'{path.text}' is not filled.")
    value = _select(value, plan.leaf_index, meta.path_string)

    # Dimensions of the data node that remain after the index
    dims = list(range(meta.ndim))
    if plan.leaf_index is not None:
        dims = dims[1:]
    time_dim = next(
        (d for d in dims if meta.coordinates[d].is_time_coordinate), None
    )
    other = [d for d in dims if d != time_dim]

    if time_dim is None:  # static data
        if not other:
            return Trace(path, None, value, value_axis, doc, fixed_time)
        if len(other) == 1:
            x, x_axis = _coordinate(leaf, other[0], len(value))
            return ProfileSeries(path, None, [x], [value], x_axis, value_axis, doc,
                                 fixed_time)
        if len(other) == 2:
            x1, a1 = _coordinate(leaf, other[0], value.shape[0])
            x2, a2 = _coordinate(leaf, other[1], value.shape[1])
            return MapSeries(path, None, 1, lambda i: (x1, x2, value), a1, a2,
                             value_axis, doc, fixed_time)
        raise PathError(f"'{path.text}' has {len(other)} dimensions; idscope can show up to 2.")

    axis_t = dims.index(time_dim)
    time = _time_of(leaf, time_dim, value.shape[axis_t])
    if not other:
        return Trace(path, time, value, value_axis, doc, fixed_time)
    moved = np.moveaxis(value, axis_t, 0)
    if len(other) == 1:
        x, x_axis = _coordinate(leaf, other[0], moved.shape[1])
        return ProfileSeries(path, time, [x] * len(time), list(moved), x_axis,
                             value_axis, doc, fixed_time)
    if len(other) == 2:
        x1, a1 = _coordinate(leaf, other[0], moved.shape[1])
        x2, a2 = _coordinate(leaf, other[1], moved.shape[2])
        return MapSeries(path, time, len(time), lambda i: (x1, x2, moved[i]), a1, a2,
                         value_axis, doc, fixed_time)
    raise PathError(
        f"'{path.text}' has {len(other)} dimensions besides time; idscope can show "
        "up to 2."
    )


def apply_x(data: SignalData, x_data: SignalData) -> SignalData:
    """Use the values of ``x_data`` as x coordinate of the profiles in ``data``.

    Both must be profiles with the same number of slices and points, for example
    ``profiles_1d/electrons/temperature`` with ``profiles_1d/grid/psi``.
    """
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
    return ProfileSeries(
        data.path, data.time, new_xs, data.ys, x_data.value_axis, data.value_axis,
        data.documentation, data.fixed_time,
    )


def _interp(grid: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y(x) on ``grid``, NaN outside of the data range."""
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    if x.size < 2:
        return np.full(len(grid), np.nan)
    order = np.argsort(x)
    return np.interp(grid, x[order], y[order], left=np.nan, right=np.nan)
