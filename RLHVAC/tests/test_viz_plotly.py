import math
from rlhvac.spec import SceneSchema, VarSpec
from rlhvac.ui import viz_plotly as viz


def _schema():
    return SceneSchema(
        color_by="temp", color_range=(10.0, 30.0), layout="grid",
        variables=[VarSpec("temp", "Temp", "C", "temperature"),
                   VarSpec("load", "Load", "kW", "power")])


def _frame(t0=20.0, t1=25.0):
    return {"step": 3, "reward": -1.0,
            "scene": {"b0": {"temp": t0, "load": 2.0}, "b1": {"temp": t1, "load": 3.0}}}


def test_heatmap_figure_colors_by_color_by_and_sets_range():
    fig = viz.heatmap_figure(_schema(), _frame())
    hm = fig.data[0]
    assert hm.type == "heatmap"
    assert hm.zmin == 10.0 and hm.zmax == 30.0
    # both unit temperatures appear somewhere in the z grid
    flat = [v for row in hm.z for v in row if v is not None]
    assert 20.0 in flat and 25.0 in flat


def test_heatmap_hover_lists_all_variables():
    fig = viz.heatmap_figure(_schema(), _frame())
    text = "".join(str(c) for row in fig.data[0].text for c in row)
    assert "Temp" in text and "Load" in text and "b0" in text


def test_variable_timeseries_one_trace_per_unit():
    frames = [_frame(20, 25), _frame(21, 26), _frame(22, 27)]
    fig = viz.variable_timeseries(frames, _schema(), "temp")
    assert len(fig.data) == 2
    ys = {tuple(tr.y) for tr in fig.data}
    assert (20, 21, 22) in ys and (25, 26, 27) in ys


def test_reward_timeseries():
    frames = [{"step": 0, "reward": -3.0}, {"step": 1, "reward": -2.0}]
    fig = viz.reward_timeseries(frames)
    assert list(fig.data[0].y) == [-3.0, -2.0]


def test_episode_bar_figure_from_summary():
    fig = viz.episode_bar_figure({"cost_total": 1.2, "carbon_total": 0.8, "note": "x"})
    bar = fig.data[0]
    assert set(bar.x) == {"cost_total", "carbon_total"}  # non-numeric 'note' dropped


def test_rollup_curve_figure():
    rollup = [{"episode": 0, "total_reward": -10.0}, {"episode": 1, "total_reward": -8.0}]
    fig = viz.rollup_curve_figure(rollup, "total_reward")
    assert list(fig.data[0].x) == [0, 1]
    assert list(fig.data[0].y) == [-10.0, -8.0]


def test_episode_bar_drops_nan():
    fig = viz.episode_bar_figure({"good": 1.0, "bad": float("nan")})
    assert list(fig.data[0].x) == ["good"]
