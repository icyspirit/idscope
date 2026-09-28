"""Parsing of idscope signal paths.

A signal path points at one quantity in an IDS, for example::

    core_profiles/profiles_1d/electrons/temperature
    equilibrium:1/time_slice/global_quantities/ip
    magnetics/flux_loop[3]/flux/data
    magnetics/flux_loop(4)/flux/data
    core_profiles/profiles_1d/electrons/temperature[0]
    pf_active/coil[*]/current/data
    pf_active/coil[*]/current/data[0]
    pf_active/coil[0:6]/element[*]/geometry
    equilibrium/time_slice/profiles_2d[0]/psi[:, 40]

Rules:

- The first element is the IDS name, optionally followed by ``:occurrence``.
- Nodes are separated by ``/`` or ``.``.
- Indices are written Python style ``[i]`` (0-based, negative from the end) or
  Fortran style ``(i)`` (1-based), as in the Data Dictionary documentation. The two
  styles cannot be mixed in one path.
- An index is a number, ``*`` or ``:`` (all elements), or a range ``a:b`` or
  ``a:b:step`` (Fortran ranges include both ends: ``(1:6)`` is ``[0:6]``).
- The time-dependent array of structures (``profiles_1d``, ``time_slice``, ...) can
  be left without index, or written ``(itime)`` as in the Data Dictionary: it is the
  time axis of the signal. A number selects one time slice, a range some of them.
  Whether an array of structures depends on time is only known from the Data
  Dictionary, so it is checked during extraction.
- Other arrays of structures need an index. ``*``, ``:`` or a range is a wildcard:
  the signal then has a member per element, e.g. one time trace per coil for
  ``pf_active/coil[*]/current/data``.
- An index applies to the node it is written on. On the last (data) node it selects
  numpy style along the dimensions of the data, in the order of the Data Dictionary
  coordinates: ``temperature[0]`` is the value on the first radial grid point,
  ``psi[:, 40]`` the 41st column of a map, and ``pf_active/coil[*]/current/data[0]``
  the first current sample of every coil.
- ``norm(path)`` normalizes every profile between its first and last point,
  ``(y - y[0]) / (y[-1] - y[0])``: ``norm(equilibrium/time_slice/profiles_1d/psi)``
  is the normalized poloidal flux psi_n, since the profiles go from the magnetic
  axis to the boundary.
- ``transpose(path)`` swaps the two dimensions of a map. Dimensions are otherwise
  never reordered: the first one is drawn along x, the second along y.
"""

import dataclasses
import re
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple, Union

from idscope.errors import PathError


class Range(NamedTuple):
    """Several elements, as a Python slice ``[start:stop:step]``."""

    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    def as_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def __str__(self) -> str:
        text = ":".join("" if v is None else str(v) for v in (self.start, self.stop))
        return text if self.step is None else f"{text}:{self.step}"


ALL = Range()
"""All elements: ``[*]``, ``[:]`` or ``(:)``."""

Selector = Union[int, Range]
Index = Optional[Selector]
"""Index of a node: ``None`` (no index), a 0-based int or a :class:`Range`."""

_HEAD = re.compile(
    r"#?(?P<ids>[a-z][a-z0-9_]*)(?::(?P<occ>\d+))?(?:(?P<sep>[/.])(?P<rest>.*))?",
    re.ASCII | re.DOTALL,
)
_NODE = re.compile(
    r"\s*(?P<name>[a-z][a-z0-9_]*)\s*(?:(?P<open>[\[(])(?P<index>[^\[\]()]*)"
    r"(?P<close>[\])]))?\s*",
    re.ASCII,
)
_GENERIC = re.compile(r"itime|i\d", re.ASCII)
"""Placeholders of the Data Dictionary documentation, e.g. ``profiles_1d(itime)``."""
_FUNCTION = re.compile(r"(?P<function>[a-z]+)\s*\((?P<inner>.*)\)", re.ASCII | re.DOTALL)

TRANSFORMS = ("norm", "transpose")
"""Functions that can be applied to a path."""


@dataclass(frozen=True)
class SignalPath:
    """A parsed signal path."""

    text: str
    """The path as written by the user (whitespace stripped)."""
    ids_name: str
    occurrence: int
    parts: Tuple[str, ...]
    """Node names below the IDS toplevel."""
    indices: Tuple[Index, ...]
    """Index of every node in :attr:`parts` (``None`` for the last one, whose index
    is :attr:`data_index`)."""
    data_index: Tuple[Selector, ...] = ()
    """Index written on the last node: one selector per dimension of the data (numpy
    style, in the order of the Data Dictionary coordinates)."""
    transform: Optional[str] = None
    """Function applied to the data: ``norm`` or ``transpose``."""

    @property
    def dd_path(self) -> str:
        """The Data Dictionary path without indices, e.g. ``profiles_1d/zeff``."""
        return "/".join(self.parts)

    def canonical(self) -> str:
        """Normalized Python-style form of the path."""
        head = self.ids_name
        if self.occurrence:
            head += f":{self.occurrence}"
        path = "/".join([head] + self._formatted())
        return f"{self.transform}({path})" if self.transform else path

    @property
    def name(self) -> str:
        """Name of the quantity, e.g. ``temperature``, or ``psi_n`` for ``norm(psi)``."""
        return f"{self.parts[-1]}_n" if self.transform == "norm" else self.parts[-1]

    @property
    def label(self) -> str:
        """A short label for legends, e.g. ``electrons/temperature``.

        The label starts at the last explicitly indexed node when there is one, so
        that ``flux_loop[3]/flux/data`` does not collapse to ``flux/data``.
        """
        if self.transform == "norm":
            return self.name
        formatted = self._formatted()
        start = max(len(formatted) - 2, 0)
        for pos, index in enumerate(self.indices):
            if isinstance(index, int):
                start = min(start, pos)
        return "/".join(formatted[start:])

    def _formatted(self) -> List[str]:
        formatted = [
            part if index is None
            else f"{part}[{'*' if index == ALL else _format_selector(index)}]"
            for part, index in zip(self.parts, self.indices)
        ]
        if self.data_index:
            formatted[-1] += f"[{', '.join(map(_format_selector, self.data_index))}]"
        return formatted

    def __str__(self) -> str:
        return self.text


def _format_selector(selector: Selector) -> str:
    return str(selector) if isinstance(selector, Range) else str(int(selector))


ALTERNATIVE_SEPARATOR = "|"


def parse_alternatives(text: str) -> List[SignalPath]:
    """Parse alternative paths ``a | b | ...``, to be tried in order.

    Alternatives let one configuration work with data that uses different names,
    e.g. ``core_profiles/profiles_1d/j_phi | core_profiles/profiles_1d/j_tor`` for
    Data Dictionary 4 and 3.

    Raises:
        PathError: when one of the paths is malformed.
    """
    parts = text.split(ALTERNATIVE_SEPARATOR)
    if len(parts) > 1 and any(not part.strip() for part in parts):
        raise PathError(f"Empty alternative in '{text.strip()}'.")
    return [parse_signal_path(part) for part in parts]


def parse_signal_path(text: str) -> SignalPath:
    """Parse a signal path.

    Raises:
        PathError: when the path is malformed.
    """
    stripped = text.strip()
    function = _FUNCTION.fullmatch(stripped)
    if function:  # an IDS name is always followed by "/", so this is a function
        name = function["function"]
        if name not in TRANSFORMS:
            raise PathError(
                f"Unknown function '{name}' in '{stripped}'; available: "
                f"{', '.join(TRANSFORMS)}."
            )
        if not function["inner"].strip():
            raise PathError(
                f"'{stripped}' needs a signal path, for example "
                f"'{name}(equilibrium/time_slice/profiles_1d/psi)'."
            )
        inner = parse_signal_path(function["inner"])
        if inner.transform:
            raise PathError(f"Functions cannot be nested: '{stripped}'.")
        return dataclasses.replace(inner, text=stripped, transform=name)
    match = _HEAD.fullmatch(stripped)
    if match is None:
        raise PathError(
            f"Invalid signal path '{text}': expected 'ids_name/path/to/quantity', "
            "for example 'core_profiles/profiles_1d/electrons/temperature'."
        )
    ids_name = match["ids"]
    occurrence = int(match["occ"]) if match["occ"] else 0
    rest = match["rest"] or ""
    if not rest:
        raise PathError(
            f"Signal path '{text}' only names the IDS '{ids_name}'; add the path "
            "to a quantity, for example 'equilibrium/time_slice/global_quantities/ip'."
        )
    parts, indices, data_index = _parse_nodes(rest, stripped)
    return SignalPath(stripped, ids_name, occurrence, parts, indices, data_index)


def _parse_nodes(rest: str, text: str):
    """Node names, their indices and the index of the last node."""
    parts: List[str] = []
    selectors: List[List[Index]] = []
    style = None
    pos = 0
    while True:
        match = _NODE.match(rest, pos)
        if match is None or match.end() == pos:
            raise PathError(
                f"Invalid signal path '{text}': cannot read a node name at "
                f"'{rest[pos:]}'."
            )
        name = match["name"]
        if match["open"]:
            node_style = "fortran" if match["open"] == "(" else "python"
            if match["close"] != {"fortran": ")", "python": "]"}[node_style]:
                raise PathError(f"Invalid signal path '{text}': unbalanced brackets.")
            if style is not None and style != node_style:
                raise PathError(
                    f"Invalid signal path '{text}': Fortran-style (1-based) and "
                    "Python-style (0-based) indices cannot be mixed."
                )
            style = node_style
            where = f"'{name}' in '{text}'"
            selectors.append([
                _parse_selector(item, node_style == "fortran", where)
                for item in match["index"].split(",")
            ])
        else:
            selectors.append([])
        parts.append(name)
        pos = match.end()
        if pos == len(rest):
            break
        if rest[pos] not in "/.":
            raise PathError(
                f"Invalid signal path '{text}': unexpected '{rest[pos]}' after "
                f"'{name}'."
            )
        pos += 1
    indices: List[Index] = []
    for name, given in zip(parts[:-1], selectors[:-1]):
        if len(given) > 1:
            raise PathError(
                f"Invalid index for '{name}' in '{text}': an array of structures "
                "takes one index; only the last node takes several."
            )
        indices.append(given[0] if given else None)
    indices.append(None)
    last = selectors[-1]
    if last and last[0] is None and len(last) == 1:
        # e.g. "profiles_1d(itime)" at the end: the path does not reach a quantity,
        # which extraction reports
        last = [None]
    if any(selector is None for selector in last) and len(last) > 1:
        raise PathError(
            f"Invalid index for '{parts[-1]}' in '{text}': placeholders such as "
            "'itime' only stand for the time array of structures."
        )
    data_index = tuple(s for s in last if s is not None)
    return tuple(parts), tuple(indices), data_index


def _parse_selector(text: str, fortran: bool, where: str) -> Index:
    item = text.strip()
    if not item:
        raise PathError(f"Empty index for {where}.")
    if item in ("*", ":"):
        return ALL
    if _GENERIC.fullmatch(item):
        return None
    if ":" in item:
        pieces = item.split(":")
        if len(pieces) > 3:
            raise PathError(f"Invalid range '{item}' for {where}.")
        numbers = [_number(p, fortran, where) if p.strip() else None for p in pieces]
        start, stop = numbers[0], numbers[1]
        step = numbers[2] if len(numbers) == 3 else None
        if step == 0:
            raise PathError(f"Invalid range '{item}' for {where}: the step is 0.")
        if fortran and start is not None:
            start -= 1  # 1-based, both ends included: (2:3) is [1:3]
        return Range(start, stop, step)
    number = _number(item, fortran, where)
    return number - 1 if fortran else number


def _number(text: str, fortran: bool, where: str) -> int:
    try:
        number = int(text.strip())
    except ValueError:
        raise PathError(
            f"Invalid index '{text.strip()}' for {where}: expected a number, '*', ':' "
            "or a range such as '0:6'."
        ) from None
    if fortran and number < 1:
        raise PathError(
            f"Invalid index for {where}: Fortran-style indices such as '(1)' start "
            "at 1."
        )
    return number
