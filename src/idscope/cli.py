"""Command line interface: ``idscope [CONFIG.yaml] [SOURCE ...]``."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from idscope import __version__
from idscope.config import ScopeConfig, default_config, load_config
from idscope.errors import ConfigError, IdscopeError
from idscope.session import config_dir, load_session
from idscope.templates import load_template, template_names

EPILOG = """\
examples:
  idscope                                continue the last session
  idscope shot.nc                        the panels of the last session, other data
  idscope --new shot.nc                  browse a netCDF file in an empty panel
  idscope scope.yaml                     open a saved configuration
  idscope scope.yaml -s A=a.nc -s B=b.nc apply a configuration to other data
  idscope -T equilibrium a.nc b.nc       compare two sources with a template
  idscope scope.yaml -e scope.png -t 5   save an image at t = 5 s, without window
  idscope --list shot.nc                 list the IDSs of a source
  idscope --list shot.nc core_profiles   list the quantities of an IDS

Sources are IDS netCDF files, IMAS URIs (imas:hdf5?path=...) or HDF5 entry
directories. Prefix a source with LABEL= to name it in legends.

Without a configuration file or a template, idscope shows the panels of the last
session. It keeps them, and its settings, in {config_dir}
($IDSCOPE_CONFIG_DIR sets another directory).
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idscope",
        description="A lightweight jScope-like viewer for IMAS IDS data.",
        epilog=EPILOG.format(config_dir=config_dir()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs", nargs="*", metavar="CONFIG_OR_SOURCE",
        help="a configuration file (.yaml) and/or data sources",
    )
    parser.add_argument(
        "-s", "--source", action="append", default=[], metavar="SOURCE",
        help="data source; replaces the sources of the configuration (repeatable)",
    )
    parser.add_argument(
        "-T", "--template", metavar="NAME",
        help="start from a template (see --list-templates) instead of a configuration",
    )
    parser.add_argument(
        "--new", action="store_true",
        help="start with an empty panel instead of the panels of the last session",
    )
    parser.add_argument(
        "--list-templates", action="store_true", help="print the template names and exit"
    )
    parser.add_argument("-t", "--time", type=float, help="initial time [s]")
    parser.add_argument(
        "--dd-version", metavar="VERSION",
        help="Data Dictionary version to open the data with (default: imas-python's)",
    )
    parser.add_argument(
        "-e", "--export", metavar="IMAGE",
        help="save the panels as an image (PNG, JPG) and exit without a window",
    )
    parser.add_argument(
        "--size", default="1400x900", metavar="WxH",
        help="window size for --export (default: 1400x900)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the IDSs of a source, or the quantities of an IDS, and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="more logging")
    parser.add_argument("--version", action="version", version=f"idscope {__version__}")
    return parser


def assemble_config(args) -> Tuple[ScopeConfig, Optional[Path], bool]:
    """Combine a configuration and the sources given on the command line.

    The configuration is a file, a template or else the last session. Returns it, its
    file and whether its panels have changes that are not saved to that file.
    """
    from idscope.data.sources import parse_source, unique_labels

    configs = [i for i in args.inputs if i.lower().endswith((".yaml", ".yml"))]
    sources = [i for i in args.inputs if i not in configs] + args.source
    if len(configs) + bool(args.template) > 1:
        raise IdscopeError("Give one configuration file or one template.")
    config_path = Path(configs[0]) if configs else None
    modified = False
    if config_path:
        config = load_config(config_path)
    elif args.template:
        config = load_template(args.template)
    else:
        session = None if args.new else _last_session()
        if session is not None:
            config, config_path, modified = session
        else:
            config = default_config()
    if sources:
        config.sources = unique_labels([parse_source(s) for s in sources])
    if args.dd_version:
        config.dd_version = args.dd_version
    if args.time is not None:
        config.time = args.time
    return config, config_path, modified


def _last_session():
    try:
        return load_session()
    except ConfigError as exc:
        print(f"idscope: warning: {exc}\nStarting with an empty panel.", file=sys.stderr)
        return None


def list_command(args) -> int:
    from idscope.data.browse import list_signals
    from idscope.data.sources import DataSource, parse_source

    if not 1 <= len(args.inputs) <= 2:
        raise IdscopeError("--list needs a source and optionally an IDS name.")
    source = DataSource(parse_source(args.inputs[0]), args.dd_version)
    try:
        if len(args.inputs) == 1:
            for name, occurrence in source.available_ids():
                print(name if occurrence == 0 else f"{name}:{occurrence}")
        else:
            name, _, occ = args.inputs[1].partition(":")
            items = list_signals(source, name, int(occ) if occ else 0)
            width = max((len(item.signal_path) for item in items), default=0)
            for item in items:
                units = f"[{item.units}]" if item.units else ""
                print(f"{item.kind:9s} {item.signal_path:{width}s} {units}")
        for note in source.notes:
            print(f"note: {note}", file=sys.stderr)
    finally:
        source.close()
    return 0


def export_command(config: ScopeConfig, config_path, args) -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from pyqtgraph.Qt import QtWidgets

    from idscope.gui import create_application, create_window

    try:
        width, height = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        raise IdscopeError(f"Invalid --size '{args.size}', expected e.g. 1400x900.")
    app = create_application()
    window = create_window(config, config_path, args.time)
    window.browser_dock.hide()
    window.resize(width, height)
    window.show()
    app.processEvents()
    window.export_image(args.export)
    for line in window.log_view.toPlainText().splitlines():
        print(f"idscope: {line}", file=sys.stderr)
    window.close()
    QtWidgets.QApplication.processEvents()
    print(f"wrote {args.export}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.list_templates:
            for name in template_names():
                print(name)
            return 0
        if args.list:
            return list_command(args)
        config, config_path, modified = assemble_config(args)
        if args.export:
            return export_command(config, config_path, args)
        try:
            from idscope.gui import run
        except ImportError as exc:
            print(
                f"idscope: error: the GUI needs a Qt binding ({exc}).\n"
                "Install one, e.g.: pip install PySide6   (or PyQt5)",
                file=sys.stderr,
            )
            return 1
        return run(config, config_path, args.time, modified)
    except IdscopeError as exc:
        print(f"idscope: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
