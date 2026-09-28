"""Exceptions raised by idscope."""


class IdscopeError(Exception):
    """Base class for all errors that idscope reports to the user."""


class PathError(IdscopeError):
    """A signal path is malformed or does not match the Data Dictionary."""


class SourceError(IdscopeError):
    """A data source cannot be opened or read."""


class NoDataError(IdscopeError):
    """The requested IDS or quantity is not filled in a data source."""


class ConfigError(IdscopeError):
    """A configuration file is invalid."""
