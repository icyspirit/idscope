from pathlib import Path

import pytest
import yaml

from idscope.config import SignalConfig, default_config, load_config
from idscope.data.sources import SourceSpec, parse_source
from idscope.errors import ConfigError
from idscope.session import config_dir, load_session, save_session, session_path


def test_config_dir(monkeypatch, tmp_path, user_config_dir):
    assert config_dir() == user_config_dir
    monkeypatch.delenv("IDSCOPE_CONFIG_DIR")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr("sys.platform", "linux")
    assert config_dir() == tmp_path / "xdg" / "idscope"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert config_dir() == Path.home() / ".config" / "idscope"


def test_round_trip(tmp_path):
    assert load_session() is None
    config = default_config(1, 2)
    config.sources = [SourceSpec(str(tmp_path / "a.nc"), "A"),
                      SourceSpec("imas:hdf5?path=/data/entry", "entry")]
    config.columns[0].panels[1].signals = [SignalConfig("pf_active/coil[:]/current/data")]
    config.time = 2.5
    scope = tmp_path / "scope.yaml"
    scope.write_text("idscope: 1\n")
    path = save_session(config, scope, modified=True)
    assert path == session_path()

    session = load_session()
    assert session.config == config
    assert session.config_path == scope
    assert session.modified
    # the session is also a configuration
    assert load_config(path) == config

    save_session(config)
    assert load_session()[1:] == (None, False)
    assert "session" not in yaml.safe_load(path.read_text())


def test_forget_a_deleted_configuration(tmp_path):
    scope = tmp_path / "scope.yaml"
    scope.write_text("idscope: 1\n")
    save_session(default_config(), scope, modified=True)
    scope.unlink()
    session = load_session()
    assert session.config_path is None
    assert session.modified


def test_sources_relative_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = default_config()
    config.sources = [parse_source("shot.nc"), parse_source("B=data/b.nc")]
    save_session(config)
    monkeypatch.chdir(Path.home())
    uris = [Path(s.uri) for s in load_session().config.sources]
    assert all(uri.is_absolute() for uri in uris)
    assert [uri.resolve() for uri in uris] == [
        (tmp_path / "shot.nc").resolve(), (tmp_path / "data" / "b.nc").resolve(),
    ]


@pytest.mark.parametrize("text", ["columns: [\n", "colums: []\n", "- a\n- b\n"])
def test_invalid_session(text):
    session_path().parent.mkdir(parents=True)
    session_path().write_text(text)
    with pytest.raises(ConfigError, match="Cannot read the last session"):
        load_session()


def test_empty_session_file():
    session_path().parent.mkdir(parents=True)
    session_path().write_text("")
    assert load_session() is None
