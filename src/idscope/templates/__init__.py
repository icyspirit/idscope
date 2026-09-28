"""Configuration templates for common IDSs.

A template is a configuration without sources: ``idscope -T equilibrium shot.nc``
applies the ``equilibrium`` template to ``shot.nc``. Templates use alternative paths
(``a | b``) so that they work with Data Dictionary 3 and 4 data.
"""

from importlib import resources
from typing import List

import yaml

from idscope.config import ScopeConfig, config_from_dict
from idscope.errors import ConfigError

_SUFFIX = ".yaml"


def template_names() -> List[str]:
    """Names of the available templates."""
    return sorted(
        entry.name[: -len(_SUFFIX)]
        for entry in resources.files(__name__).iterdir()
        if entry.name.endswith(_SUFFIX)
    )


def template_text(name: str) -> str:
    """The YAML text of a template."""
    entry = resources.files(__name__) / f"{name}{_SUFFIX}"
    if not entry.is_file():
        raise ConfigError(
            f"Unknown template '{name}'; available: {', '.join(template_names())}."
        )
    return entry.read_text()


def load_template(name: str) -> ScopeConfig:
    """Load a template as a configuration (without sources)."""
    try:
        return config_from_dict(yaml.safe_load(template_text(name)))
    except ConfigError as exc:
        raise ConfigError(f"Template '{name}': {exc}") from None
