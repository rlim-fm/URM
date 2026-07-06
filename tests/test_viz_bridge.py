"""Integration tests for viz_bridge.py, the glue between URM/TARM's
training loop and Tropical-RNN's streaming Visualizer.
"""
import os
import sys

import pytest

import viz_bridge
from viz_bridge import PuzzleTrainingLog, PuzzleVisualizerBridge, _extract_loss


def test_extract_loss_flat_train_metrics():
    metrics = {"train/lm_loss": 1.23, "train/q_halt_loss": 4.0}
    assert _extract_loss(metrics) == pytest.approx(1.23)


def test_extract_loss_nested_eval_metrics():
    metrics = {"test_set": {"lm_loss": 2.5, "accuracy": 0.9}}
    assert _extract_loss(metrics) == pytest.approx(2.5)


def test_extract_loss_missing_key_returns_none():
    metrics = {"train/accuracy": 0.9, "train/q_halt_loss": 1.0}
    assert _extract_loss(metrics) is None


def test_extract_loss_empty_dict_returns_none():
    assert _extract_loss({}) is None
    assert _extract_loss(None) is None


def test_puzzle_training_log_initial_state():
    log = PuzzleTrainingLog()
    assert log.logs == {"train_loss": [], "test_loss": []}
    assert log.metadata == {}


def test_visualizer_bridge_log_train_and_eval_append(tmp_path):
    bridge = PuzzleVisualizerBridge(output_dir=str(tmp_path), name="test_run")

    bridge.log_train({"train/lm_loss": 1.0}, epoch=0)
    bridge.log_train({"train/lm_loss": 0.5}, epoch=1)
    bridge.log_eval({"test": {"lm_loss": 0.8}}, epoch=1)

    assert bridge._log.logs["train_loss"] == [1.0, 0.5]
    assert bridge._log.logs["test_loss"] == [0.8]


def test_visualizer_bridge_log_train_ignores_missing_loss(tmp_path):
    bridge = PuzzleVisualizerBridge(output_dir=str(tmp_path), name="test_run")
    bridge.log_train({"train/accuracy": 0.9}, epoch=0)
    assert bridge._log.logs["train_loss"] == []


def test_visualizer_bridge_finalize_produces_plot(tmp_path):
    bridge = PuzzleVisualizerBridge(output_dir=str(tmp_path), name="test_run")
    for epoch, loss in enumerate([1.0, 0.7, 0.4]):
        bridge.log_train({"train/lm_loss": loss}, epoch=epoch)
        bridge.log_eval({"test": {"lm_loss": loss + 0.1}}, epoch=epoch)

    bridge.finalize()
    bridge._visualizer.wait_for_background(timeout=30)

    output_file = tmp_path / "test_run_loss_history.png"
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


def test_viz_bridge_isolation_does_not_pollute_sys_modules():
    """viz_bridge loads Tropical-RNN/visualization.py under a private module
    name specifically so it never registers top-level `models`/`train`/
    `datasets` modules that would collide with URM's own packages. Guard
    against that isolation regressing.
    """
    # viz_bridge loads the submodule via module_from_spec/exec_module without
    # ever registering it in sys.modules, so no top-level name is claimed at all.
    assert "_tropical_rnn_visualization" not in sys.modules
    assert not any(name in sys.modules for name in ("train", "datasets"))

    # URM's own `models` package (imported all over this repo) must be the
    # one actually bound in sys.modules, not something Tropical-RNN's
    # visualization.py substituted in.
    import models as urm_models
    assert sys.modules["models"] is urm_models
    urm_models_dir = os.path.dirname(list(urm_models.__path__)[0])
    assert urm_models_dir == os.path.dirname(os.path.abspath(viz_bridge.__file__))
