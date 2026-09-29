"""Signal expressions: arithmetic on signals.

A signal can be computed from others and from numbers, e.g.::

    equilibrium/time_slice/global_quantities/ip - summary/global_quantities/ip/value
    core_profiles/profiles_1d/t_i_average / core_profiles/profiles_1d/electrons/temperature
    deriv(core_profiles/global_quantities/ip)
    abs(pf_active/coil[:]/current/data) * 1e-3

Rules:

- The operators ``+``, ``-``, ``*`` and ``/`` combine signals and numbers, with the
  usual precedence and parentheses. Since ``/`` also separates the nodes of a path,
  a division is written with spaces around it: ``a / b``, while ``a/b`` is a path
  (a ``/`` right after a path, with no space, belongs to the path).
- Functions: ``deriv(x)`` is the derivative along time for time traces and along
  the coordinate for profiles, ``abs(x)`` the absolute value, and ``norm(x)``
  normalizes profiles (see :mod:`idscope.data.paths`, also for ``transpose``).
- Two signals are combined on the times and the grid of the first: the second is
  interpolated in time for traces, and taken at its closest time slice (within its
  time range) and interpolated on the grid of the first for profiles and maps. A
  time trace combined with profiles or maps applies to each slice at its time,
  e.g. a profile divided by its central value.
- A signal with wildcards (e.g. a trace per coil) is combined member by member with
  another one, by name, or with the same signal for all its members.

The same arithmetic compares data sources in a panel: :func:`relative_to` gives the
difference or the ratio of the data of two sources.
"""

import operator
import re
from typing import Callable, List, Optional, Union

import numpy as np

from idscope.data.extract import (
    Axis,
    Channels,
    MapSeries,
    ProfileSeries,
    Shapes,
    SignalData,
    Trace,
    _interp,
    normalize,
)
from idscope.data.paths import SignalPath, _split_arguments, parse_signal_path
from idscope.errors import PathError

FUNCTIONS = ("deriv", "abs", "norm", "transpose")
"""Functions that expressions can use."""

_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul,
              "/": operator.truediv}
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
_NUMBER = re.compile(r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_NAME = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_CLOSING = {"[": "]", "(": ")"}

Value = Union[SignalData, float]


# ---------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------


class _Node:
    def canonical(self) -> str:
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError

    def paths(self) -> List[SignalPath]:
        return []

    def evaluate(self, read: Callable[[SignalPath], SignalData]) -> Value:
        raise NotImplementedError


class _Path(_Node):
    def __init__(self, path: SignalPath):
        self.path = path

    def canonical(self) -> str:
        return self.path.canonical()

    def label(self) -> str:
        return self.path.label

    def paths(self) -> List[SignalPath]:
        return [self.path]

    def evaluate(self, read):
        return read(self.path)


class _Number(_Node):
    def __init__(self, value: float, text: str):
        self.value, self.text = value, text

    def canonical(self) -> str:
        return repr(self.value)

    def label(self) -> str:
        return self.text

    def evaluate(self, read):
        return self.value


class _Negate(_Node):
    def __init__(self, operand: _Node):
        self.operand = operand

    def canonical(self) -> str:
        return f"(-{self.operand.canonical()})"

    def label(self) -> str:
        inner = self.operand.label()
        return f"-({inner})" if isinstance(self.operand, _Binary) else f"-{inner}"

    def paths(self):
        return self.operand.paths()

    def evaluate(self, read):
        value = self.operand.evaluate(read)
        return -value if isinstance(value, float) else _elementwise(
            value, np.negative, Axis(f"-{value.value_axis.label}", value.value_axis.units))


class _Binary(_Node):
    def __init__(self, op: str, left: _Node, right: _Node):
        self.op, self.left, self.right = op, left, right

    def canonical(self) -> str:
        return f"({self.left.canonical()} {self.op} {self.right.canonical()})"

    def label(self) -> str:
        def side(node, right):
            text = node.label()
            if isinstance(node, _Binary) and (
                    _PRECEDENCE[node.op] < _PRECEDENCE[self.op]
                    or (right and _PRECEDENCE[node.op] == _PRECEDENCE[self.op]
                        and self.op in "-/")):
                return f"({text})"
            return text

        return f"{side(self.left, False)} {self.op} {side(self.right, True)}"

    def paths(self):
        return self.left.paths() + self.right.paths()

    def evaluate(self, read):
        return combine(self.op, self.left.evaluate(read), self.right.evaluate(read))


class _Call(_Node):
    def __init__(self, function: str, argument: _Node):
        self.function, self.argument = function, argument

    def canonical(self) -> str:
        return f"{self.function}({self.argument.canonical()})"

    def label(self) -> str:
        return f"{self.function}({self.argument.label()})"

    def paths(self):
        return self.argument.paths()

    def evaluate(self, read):
        value = self.argument.evaluate(read)
        if isinstance(value, float):
            raise PathError(f"{self.function}() applies to a signal, not to a number.")
        if self.function == "deriv":
            return derivative(value)
        if self.function == "abs":
            return _elementwise(value, np.abs, Axis(f"abs({value.value_axis.label})",
                                                    value.value_axis.units))
        return normalize(value)  # norm


class Expression:
    """A signal computed from signal paths and numbers (see the module
    documentation). It stands where a :class:`SignalPath` does: evaluated by a data
    source, it gives :class:`SignalData` whose ``path`` is the expression."""

    def __init__(self, text: str, node: _Node):
        self.text = text
        self._node = node

    def canonical(self) -> str:
        return self._node.canonical()

    @property
    def label(self) -> str:
        """A label for legends, from the labels of its paths."""
        return self._node.label()

    @property
    def name(self) -> str:
        return self.label

    def paths(self) -> List[SignalPath]:
        """The signal paths it reads."""
        return self._node.paths()

    def evaluate(self, read: Callable[[SignalPath], SignalData]) -> SignalData:
        """The signal, reading the signal of each path with ``read``."""
        value = self._node.evaluate(read)
        if isinstance(value, float):
            raise PathError(f"'{self.text}' is a number, not a signal.")
        return _renamed(value, self)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Expression({self.text!r})"


def parse_signal(text: str) -> Union[SignalPath, Expression]:
    """Parse a signal: a path (as :func:`parse_signal_path`) or an expression.

    Raises:
        PathError: when it is neither.
    """
    stripped = text.strip()
    try:
        node = _Parser(stripped).parse()
    except PathError as expression_error:
        try:  # e.g. a path written with spaces around its "/"
            return parse_signal_path(stripped)
        except PathError as path_error:
            chosen = expression_error if _looks_like_expression(stripped) else path_error
        raise chosen from None
    if isinstance(node, _Path):
        return node.path
    return Expression(stripped, node)


def _looks_like_expression(text: str) -> bool:
    """Whether a text that is neither a path nor an expression was meant as an
    expression: it starts with a function, or has an operator outside the indices
    of its paths."""
    outside = re.sub(r"\[[^\[\]]*\]", "", text)
    return bool(re.match(r"\s*[a-z]\w*\s*\(", text)
                or re.search(r"[+*]|(?:^|[\s(])-|\s/\s", outside))


class _Parser:
    """Recursive descent: sum = product (('+' | '-') product)*, product = unary
    (('*' | '/') unary)*, unary = '-' unary | atom, atom = number | path |
    function '(' sum ')' | '(' sum ')'."""

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.path_end = -1  # where the last path ends: a "/" right after it is its own

    def error(self, message: str) -> PathError:
        return PathError(f"Invalid expression '{self.text}': {message}.")

    def parse(self) -> _Node:
        if not self.text:
            raise self.error("it is empty")
        node = self.sum()
        self.skip()
        if self.pos != len(self.text):
            raise self.error(f"unexpected '{self.text[self.pos:]}'")
        return node

    def skip(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def peek(self) -> str:
        self.skip()
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def sum(self) -> _Node:
        node = self.product()
        while self.peek() in ("+", "-"):
            op = self.text[self.pos]
            self.pos += 1
            node = _Binary(op, node, self.product())
        return node

    def product(self) -> _Node:
        node = self.unary()
        while self.peek() == "*" or (self.peek() == "/" and self.pos != self.path_end):
            op = self.text[self.pos]
            self.pos += 1
            node = _Binary(op, node, self.unary())
        return node

    def unary(self) -> _Node:
        char = self.peek()
        if char == "-":
            self.pos += 1
            return _Negate(self.unary())
        if char == "+":
            self.pos += 1
            return self.unary()
        return self.atom()

    def atom(self) -> _Node:
        char = self.peek()
        if char == "(":
            self.pos += 1
            node = self.sum()
            self.expect(")")
            return node
        number = _NUMBER.match(self.text, self.pos)
        if number:
            self.pos = number.end()
            return _Number(float(number.group()), number.group())
        name = _NAME.match(self.text, self.pos + (char == "#"))
        if not name:
            found = f"'{self.text[self.pos:]}'" if char else "the end"
            raise self.error(f"expected a signal path, a number or '(' at {found}")
        after = name.end()
        while after < len(self.text) and self.text[after].isspace():
            after += 1
        if after < len(self.text) and self.text[after] == "(" and char != "#":
            return self.call(name.group(), after)
        return self.path()

    def expect(self, char: str) -> None:
        if self.peek() != char:
            found = f"'{self.text[self.pos:]}'" if self.pos < len(self.text) else "the end"
            raise self.error(f"expected '{char}' at {found}")
        self.pos += 1

    def call(self, function: str, opening: int) -> _Node:
        if function not in FUNCTIONS:
            raise self.error(f"unknown function '{function}' (available: "
                             f"{', '.join(FUNCTIONS)})")
        start = self.pos
        closing = self.closing(opening)
        if function in ("norm", "transpose"):
            # Of a path, a function of the path itself (as transpose has to be)
            arguments = _split_arguments(self.text[opening + 1: closing - 1])
            argument = _Parser(arguments[0].strip()).parse()
            if isinstance(argument, _Path):
                self.pos = closing
                return _Path(parse_signal_path(self.text[start:closing]))
            if function == "transpose":
                raise self.error("transpose() takes a signal path, not an expression")
            if len(arguments) > 1:
                raise self.error("norm() takes one argument")
            self.pos = closing
            return _Call(function, argument)
        self.pos = opening + 1
        argument = self.sum()
        self.expect(")")
        return _Call(function, argument)

    def path(self) -> _Node:
        """A path: the IDS name, its occurrence, and nodes joined by '/' or '.' with
        no space, each with an index right after its name."""
        start = self.pos
        pos = self.pos + (self.text[self.pos] == "#")
        pos = _NAME.match(self.text, pos).end()
        occurrence = re.compile(r":\d+").match(self.text, pos)
        if occurrence:
            pos = occurrence.end()
        while pos < len(self.text) and self.text[pos] in "/.":
            name = _NAME.match(self.text, pos + 1)
            if not name:
                break
            pos = name.end()
            if pos < len(self.text) and self.text[pos] in _CLOSING:
                pos = self.closing(pos)
        self.pos = self.path_end = pos
        return _Path(parse_signal_path(self.text[start:pos]))

    def closing(self, opening: int) -> int:
        """The position after the bracket that closes the one at ``opening``."""
        depth = []
        for pos in range(opening, len(self.text)):
            char = self.text[pos]
            if char in _CLOSING:
                depth.append(_CLOSING[char])
            elif char in ")]":
                if not depth or depth.pop() != char:
                    break
                if not depth:
                    return pos + 1
        raise self.error("unbalanced brackets")


# ---------------------------------------------------------------------------------
# Arithmetic on signals
# ---------------------------------------------------------------------------------


def combine(op: str, a: Value, b: Value, label: Optional[str] = None) -> Value:
    """``a op b`` for signals and numbers (``op`` is one of ``+ - * /``).

    The result is on the times and the grid of ``a`` (of ``b`` when ``a`` is a
    number or does not depend on time and ``b`` does); its values are labeled
    ``label`` (by default, the labels of ``a`` and ``b`` with the operator).

    Raises:
        PathError: when the two cannot be combined (shapes, maps with profiles,
            profiles against different coordinates, ...).
    """
    function = _OPERATORS[op]
    if isinstance(a, float) and isinstance(b, float):
        return function(a, b)
    for value in (a, b):
        if isinstance(value, Shapes):
            raise PathError(f"'{value.path.text}' is a shape; shapes cannot be computed "
                            "with.")
    axis = Axis(label or f"{_label(a)} {op} {_label(b)}", _units(op, a, b))
    if isinstance(a, Channels) or isinstance(b, Channels):
        return _members(op, a, b, axis)
    if isinstance(a, float) or isinstance(b, float):
        number = a if isinstance(a, float) else b
        data = b if isinstance(a, float) else a

        def with_number(values):
            with np.errstate(divide="ignore", invalid="ignore"):
                return function(number, values) if data is b else function(values, number)

        return _elementwise(data, with_number, axis)
    kinds = (type(a), type(b))
    if kinds == (Trace, Trace):
        return _traces(function, a, b, axis)
    if isinstance(a, MapSeries) or isinstance(b, MapSeries):
        return _maps(function, a, b, axis)
    return _profiles(function, a, b, axis)


def derivative(data: SignalData) -> SignalData:
    """d/dt of a time trace, d/dx of every profile (member by member)."""
    label = f"deriv({data.value_axis.label})"
    if isinstance(data, Channels):
        members = [derivative(member) for member in data.members]
        return Channels(data.path, members, data.names, members[0].value_axis,
                        data.documentation, data.axis_label)
    x_units = data.x_axis.units if hasattr(data, "x_axis") else ""
    axis = Axis(label, _ratio(data.value_axis.units, x_units))
    if isinstance(data, Trace):
        if data.time is None or data.values.size < 2:
            raise PathError(f"deriv() needs a time trace; '{data.path.text}' has "
                            "one value.")
        return Trace(data.path, data.time, _gradient(data.values, data.time), axis,
                     data.documentation, data.fixed_time)
    if isinstance(data, ProfileSeries):
        ys = [_gradient(y, x) for x, y in zip(data.xs, data.ys)]
        return ProfileSeries(data.path, data.time, data.xs, ys, data.x_axis, axis,
                             data.documentation, data.fixed_time)
    raise PathError(f"deriv() applies to time traces and profiles, but "
                    f"'{data.path.text}' is a {data.kind}.")


def _gradient(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """dy/dx along x (in the order of x), NaN where it cannot be told."""
    result = np.full(len(y), np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return result
    order = np.flatnonzero(valid)[np.argsort(x[valid], kind="stable")]
    xs, ys = x[order], y[order]
    keep = np.concatenate([[True], np.diff(xs) > 0])  # the same x twice: once
    order, xs, ys = order[keep], xs[keep], ys[keep]
    if len(xs) >= 2:
        result[order] = np.gradient(ys, xs)
    return result


def _label(value: Value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    label = value.value_axis.label
    return f"({label})" if " " in label and not label.startswith(("abs(", "deriv(")) \
        else label


def _unit(units: str) -> str:
    """Units, empty for dimensionless ones ("1" and "-" in the Data Dictionary)."""
    return "" if units in ("1", "-") else units


def _ratio(numerator: str, denominator: str) -> str:
    numerator, denominator = _unit(numerator), _unit(denominator)
    if not denominator:
        return numerator
    if not numerator:
        return f"{denominator}^-1"
    return f"{numerator}/{denominator}"


def _units(op: str, a: Value, b: Value) -> str:
    ua = "" if isinstance(a, float) else _unit(a.value_axis.units)
    ub = "" if isinstance(b, float) else _unit(b.value_axis.units)
    if op in "+-":
        if isinstance(b, float) or ua == ub:
            return ua
        return ub if isinstance(a, float) else ""
    if op == "*":
        return ".".join(unit for unit in (ua, ub) if unit)
    if isinstance(b, float):
        return ua
    if ua == ub:
        return ""
    return _ratio(ua, ub)


def _elementwise(data: SignalData, function, axis: Axis) -> SignalData:
    """``function`` applied to the values of every slice (and member)."""
    if isinstance(data, Channels):
        members = [_elementwise(member, function, axis) for member in data.members]
        return Channels(data.path, members, data.names, axis, data.documentation,
                        data.axis_label)
    if isinstance(data, Trace):
        return Trace(data.path, data.time, function(data.values), axis,
                     data.documentation, data.fixed_time)
    if isinstance(data, ProfileSeries):
        return ProfileSeries(data.path, data.time, data.xs, [function(y) for y in data.ys],
                             data.x_axis, axis, data.documentation, data.fixed_time)
    if isinstance(data, MapSeries):
        def load(index):
            x1, x2, z = data.at_index(index)
            return x1, x2, function(z)

        return MapSeries(data.path, data.time, data.count, load, data.x_axis,
                         data.x2_axis, axis, data.documentation, data.fixed_time)
    raise PathError(f"'{data.path.text}' is a {data.kind}; it cannot be computed with.")


def _members(op: str, a: Value, b: Value, axis: Axis) -> Channels:
    """Member by member: by name, else in order when there are as many."""
    template = a if isinstance(a, Channels) else b
    names = template.names
    if isinstance(a, Channels) and isinstance(b, Channels):
        if a.names == b.names or (len(a.names) == len(b.names)
                                  and not set(a.names) & set(b.names)):
            pairs = list(zip(a.members, b.members))  # the same, or all renamed
        else:  # those of the first that the second has too
            others = dict(zip(b.names, b.members))
            names = [name for name in a.names if name in others]
            if not names:
                raise PathError(
                    f"'{a.path.text}' and '{b.path.text}' have no element of the same "
                    "name: they cannot be combined one by one.")
            pairs = [(member, others[name]) for member, name in zip(a.members, a.names)
                     if name in others]
    elif isinstance(a, Channels):
        pairs = [(member, b) for member in a.members]
    else:
        pairs = [(a, member) for member in b.members]
    members = [combine(op, x, y, axis.label) for x, y in pairs]
    return Channels(template.path, members, names, axis, template.documentation,
                    template.axis_label)


def _traces(function, a: Trace, b: Trace, axis: Axis) -> Trace:
    base = a if a.time is not None or b.time is None else b
    with np.errstate(divide="ignore", invalid="ignore"):
        values = function(_at_times(a, base.time), _at_times(b, base.time))
    return Trace(base.path, base.time, values, axis, base.documentation,
                 base.fixed_time)


def _at_times(trace: Trace, times: Optional[np.ndarray]) -> np.ndarray:
    """The values of a trace at ``times`` (NaN outside of its time range; a lone
    sample only at its time)."""
    values = np.atleast_1d(trace.values)
    if trace.time is None:
        return values[0] if times is None else np.full(len(times), values[0])
    if times is None:
        return values
    times = np.asarray(times, dtype=float)
    if len(trace.time) == len(times) and np.array_equal(trace.time, times):
        return values
    if len(trace.time) == 1:
        return np.where(_same_time(times, trace.time[0]), values[0], np.nan)
    return _interp(times, trace.time, values)


def _same_time(t, reference: float):
    return np.isclose(t, reference, rtol=1e-9, atol=1e-12)


def _base(a: SignalData, b: SignalData, kind) -> tuple:
    """The signal of type ``kind`` whose grid the result is on, and the times of
    the result: those of the first signal, unless only the second depends on time
    (e.g. a static profile times a time trace)."""
    base = a if isinstance(a, kind) else b
    other = b if base is a else a
    if base.time is None and other.time is not None:
        if isinstance(other, kind):
            base = other
        return base, other.time
    return base, base.time


def _slice_near(data: SignalData, t: Optional[float]) -> Optional[int]:
    """The slice of ``data`` closest to ``t``, None when ``t`` is out of its time
    range (by more than half a step; a single slice only at its time)."""
    if data.time is None or t is None:
        return 0
    index = data.time_index(t)
    if index is None:
        return None
    times = np.sort(data.time[np.isfinite(data.time)])
    if len(times) == 1:
        return index if _same_time(t, times[0]) else None
    step = float(np.median(np.diff(times)))
    if t < times[0] - step / 2 or t > times[-1] + step / 2:
        return None
    return index


def _profiles(function, a: SignalData, b: SignalData, axis: Axis) -> ProfileSeries:
    """Profiles with profiles, or with the value of a trace at the time of each."""
    if isinstance(a, ProfileSeries) and isinstance(b, ProfileSeries) and \
            a.x_axis.label != b.x_axis.label:
        raise PathError(
            f"'{a.path.text}' and '{b.path.text}' are profiles against "
            f"{a.x_axis.label or 'their index'} and {b.x_axis.label or 'their index'}; "
            "they cannot be combined.")
    base, time = _base(a, b, ProfileSeries)
    times = [None] if time is None else [float(t) for t in time]
    xs, ys = [], []
    for t in times:
        index = 0 if base.time is None else _slice_near(base, t)
        x = base.xs[index] if index is not None and index < len(base.xs) else np.empty(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ys.append(function(_profile_at(a, t, x), _profile_at(b, t, x)))
        xs.append(x)
    return ProfileSeries(base.path, time, xs, ys, base.x_axis, axis,
                         base.documentation, base.fixed_time)


def _profile_at(data: SignalData, t: Optional[float], x: np.ndarray) -> np.ndarray:
    """A profile (or the value of a trace) at time ``t``, on the grid ``x``."""
    if isinstance(data, Trace):
        value = _at_times(data, None if t is None else np.array([t]))
        return np.full(len(x), float(np.atleast_1d(value)[0]))
    index = _slice_near(data, t)
    if index is None or index >= len(data.ys):
        return np.full(len(x), np.nan)
    own_x, y = data.xs[index], data.ys[index]
    if len(own_x) == len(x) and np.array_equal(own_x, x):
        return y
    return _interp(x, own_x, y)


def _maps(function, a: SignalData, b: SignalData, axis: Axis) -> MapSeries:
    """Maps with maps (on the grid of the first), or with the value of a trace at
    the time of each."""
    for data in (a, b):
        if not isinstance(data, (MapSeries, Trace)):
            raise PathError(f"'{a.path.text}' and '{b.path.text}' cannot be combined: "
                            "a map combines with maps, numbers and time traces.")
    base, time = _base(a, b, MapSeries)

    def load(index):
        t = None if time is None else float(time[index])
        x1, x2, _ = base.at_index(0 if base.time is None else _slice_near(base, t))
        with np.errstate(divide="ignore", invalid="ignore"):
            return x1, x2, function(_map_at(a, t, x1, x2), _map_at(b, t, x1, x2))

    count = base.count if time is base.time else len(time)
    return MapSeries(base.path, time, count, load, base.x_axis, base.x2_axis, axis,
                     base.documentation, base.fixed_time)


def _map_at(data: SignalData, t, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    if isinstance(data, Trace):
        value = _at_times(data, None if t is None else np.array([t]))
        return np.full((len(x1), len(x2)), float(np.atleast_1d(value)[0]))
    index = _slice_near(data, t)
    if index is None:
        return np.full((len(x1), len(x2)), np.nan)
    own1, own2, z = data.at_index(index)
    if own1.shape == x1.shape and own2.shape == x2.shape and \
            np.array_equal(own1, x1) and np.array_equal(own2, x2):
        return z
    # On the grid of the other: along x2 for every row, then along x1
    rows = np.array([_interp(x2, own2, row) for row in np.asarray(z, float)])
    if rows.size == 0:
        return np.full((len(x1), len(x2)), np.nan)
    return np.array([_interp(x1, own1, column) for column in rows.T]).T


def _renamed(data: SignalData, path) -> SignalData:
    """``data`` as the signal of ``path`` (an expression), and its members too."""
    renamed = _shallow_copy(data)
    renamed.path = path
    if isinstance(data, Channels):
        renamed.members = [_renamed(member, path) for member in data.members]
    renamed.value_axis = Axis(path.label, data.value_axis.units)
    return renamed


def _shallow_copy(data: SignalData) -> SignalData:
    copy = object.__new__(type(data))
    copy.__dict__.update(data.__dict__)
    return copy


# ---------------------------------------------------------------------------------
# Comparing sources
# ---------------------------------------------------------------------------------

RELATIVE = {"difference": "-", "ratio": "/"}
"""How a panel can show the data of a source relative to the first one."""


def relative_to(data: SignalData, reference: SignalData, relative: str) -> SignalData:
    """The data of a source relative to that of the reference source: their
    difference (``data - reference``) or their ratio (``data / reference``), on the
    times and grid of ``data``.

    Raises:
        PathError: when they cannot be combined (e.g. shapes).
    """
    label = data.value_axis.label
    return combine(RELATIVE[relative], data, reference,
                   f"Δ {label}" if relative == "difference" else f"{label} ratio")
