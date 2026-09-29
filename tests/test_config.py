import pytest

from idscope.config import (
    ColumnConfig,
    PanelConfig,
    ScopeConfig,
    SignalConfig,
    config_from_dict,
    config_to_dict,
    default_config,
    load_config,
    save_config,
)
from idscope.data.sources import SourceSpec
from idscope.demo import DEMO_CONFIG
from idscope.errors import ConfigError


def test_short_and_long_forms(tmp_path):
    config = config_from_dict(
        {
            "sources": ["A=a.nc", {"uri": "imas:hdf5?path=/d/e", "label": "ref"}],
            "columns": [
                [{"signals": ["core_profiles/global_quantities/ip"]}],
                {
                    "width": 2,
                    "panels": [
                        {
                            "title": "Te",
                            "kind": "profile",
                            "signals": [
                                {
                                    "path": "core_profiles/profiles_1d/electrons/temperature",
                                    "scale": 1e-3,
                                    "sources": "A",
                                }
                            ],
                            "y_range": [0, 10],
                        }
                    ],
                },
            ],
        },
        base_dir=tmp_path,
    )
    assert config.sources == [
        SourceSpec(str(tmp_path / "a.nc"), "A"),
        SourceSpec("imas:hdf5?path=/d/e", "ref"),
    ]
    first, second = config.columns
    assert first.panels[0].signals == [SignalConfig("core_profiles/global_quantities/ip")]
    assert second.width == 2.0
    te = second.panels[0].signals[0]
    assert te.scale == 1e-3 and te.sources == ["A"]
    assert second.panels[0].y_range == (0.0, 10.0)


def test_round_trip(tmp_path):
    config = ScopeConfig(
        title="test",
        sources=[SourceSpec(str(tmp_path / "a.nc"), "A"), SourceSpec("imas:hdf5?path=/x/y", "y")],
        columns=[
            ColumnConfig(
                [
                    PanelConfig(
                        title="Ip",
                        signals=[
                            SignalConfig("equilibrium/time_slice/global_quantities/ip"),
                            SignalConfig("summary/global_quantities/ip/value", label="summary",
                                         scale=1e-6, color="#ff0000"),
                        ],
                        x_range=(0.0, 5.0),
                        log_y=True,
                    )
                ],
                width=1.5,
            ),
            ColumnConfig([PanelConfig(kind="spacetime", height=2.0),
                          PanelConfig(kind="map", contours=12)]),
        ],
        time=2.5,
        share_x=False,
    )
    path = tmp_path / "scope.yaml"
    save_config(config, path)
    text = path.read_text()
    assert "A=a.nc" in text  # relative to the configuration file
    assert "- equilibrium/time_slice/global_quantities/ip" in text  # short form
    assert load_config(path) == config


def test_default_values_are_omitted():
    data = config_to_dict(default_config(1, 2))
    assert data == {"idscope": 1, "columns": [{"panels": [{}, {}]}]}


@pytest.mark.parametrize(
    "data, message",
    [
        ({"colums": []}, "did you mean 'columns'"),
        ({"columns": [{"panels": [{"kind": "bar"}]}]}, "'kind' must be one of"),
        ({"columns": [{"panels": [{"signals": [{"scale": 2}]}]}]}, "'path' is required"),
        ({"columns": [{"panels": [{"x_range": [1]}]}]}, r"columns\[0\].panels\[0\]"),
        ({"columns": [{"panels": [{"log_y": "yes"}]}]}, "log_y"),
        ({"idscope": 99}, "format version"),
        ({"time": True}, "time"),
        ({"columns": [{"panels": [{"contours": -1}]}]}, "must not be negative"),
        ({"columns": [{"panels": [{"contours": 2.5}]}]}, "contours"),
        ({"columns": [{"panels": [{"relative": "minus"}]}]},
         "'relative' must be one of difference, ratio"),
    ],
)
def test_invalid(data, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(data)


def test_demo_config(tmp_path):
    path = tmp_path / "demo.yaml"
    path.write_text(DEMO_CONFIG)
    config = load_config(path)
    assert [s.label for s in config.sources] == ["A", "B"]
    assert config.sources[0].uri == str(tmp_path / "demo_A.nc")
    assert len(config.panels()) == 7


def test_relative_sources(data_dir):
    path = data_dir / "relative.yaml"
    path.write_text("sources: [h5, B=b.nc]\n")
    config = load_config(path)
    assert config.sources == [
        SourceSpec(f"imas:hdf5?path={(data_dir / 'h5').resolve()}", "h5"),
        SourceSpec(str(data_dir.resolve() / "b.nc"), "B"),
    ]


def test_load_errors(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("columns: [\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(bad)


def test_default_layout_is_a_single_panel():
    config = default_config()
    assert len(config.columns) == 1 and len(config.panels()) == 1


def test_sources_relative_to_the_first(tmp_path):
    config = config_from_dict({"columns": [{"panels": [
        {"relative": "difference", "signals": ["summary/global_quantities/ip/value - "
                                                "equilibrium/time_slice/global_quantities/ip"]},
        {"relative": "ratio"}, {}]}]})
    assert [panel.relative for panel in config.panels()] == ["difference", "ratio", None]
    path = tmp_path / "relative.yaml"
    save_config(config, path)
    assert "relative: difference" in path.read_text()
    again = load_config(path)
    assert [panel.relative for panel in again.panels()] == ["difference", "ratio", None]
    assert again.panels()[0].signals[0].path.endswith(" - equilibrium/time_slice/"
                                                      "global_quantities/ip")
