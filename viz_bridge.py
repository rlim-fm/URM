"""Bridge between URM/TARM's real puzzle training loop (pretrain.py) and
Tropical-RNN's streaming Visualizer.

Loads Tropical-RNN/visualization.py in isolation (via importlib, not a normal
package import) so it never registers top-level modules named `models`,
`train`, or `datasets` that would collide with URM's own `models/` package.
"""
import importlib.util
import os

_VIS_PATH = os.path.join(os.path.dirname(__file__), "Tropical-RNN", "visualization.py")
_spec = importlib.util.spec_from_file_location("_tropical_rnn_visualization", _VIS_PATH)
_tropical_rnn_visualization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tropical_rnn_visualization)

Visualizer = _tropical_rnn_visualization.Visualizer
LossHistoryPlot = _tropical_rnn_visualization.LossHistoryPlot


class PuzzleTrainingLog:
    """Minimal stand-in for Tropical-RNN's Processor, satisfying the
    `.logs` / `.metadata` contract read by `Visualization.update(processor, epoch)`.
    """

    def __init__(self):
        self.logs = {"train_loss": [], "test_loss": []}
        self.metadata = {}


def _extract_loss(metrics: dict):
    """Pull a loss scalar out of a wandb.log-shaped metrics dict.

    Train metrics are flat (`train/lm_loss`, ...); eval metrics are nested
    (`{set_name: {"lm_loss": ..., ...}}`). Either way, take the first key
    ending in `lm_loss`.
    """
    if not metrics:
        return None
    for key, value in metrics.items():
        if isinstance(value, dict):
            nested = _extract_loss(value)
            if nested is not None:
                return nested
        elif key.endswith("lm_loss"):
            return float(value)
    return None


class PuzzleVisualizerBridge:
    """Streams URM/TARM's real train/eval metrics into a Tropical-RNN
    Visualizer, producing a loss-history plot alongside a normal training run.
    """

    def __init__(self, output_dir: str, name: str):
        self._log = PuzzleTrainingLog()
        self._visualizer = Visualizer(name=name or "urm", output_dir=output_dir)
        self._visualizer.register(LossHistoryPlot())
        self._visualizer.attach_processor(self._log)

    def log_train(self, metrics: dict, epoch: int):
        loss = _extract_loss(metrics)
        if loss is not None:
            self._log.logs["train_loss"].append(loss)
        self._visualizer.update(epoch)

    def log_eval(self, metrics: dict, epoch: int):
        loss = _extract_loss(metrics)
        if loss is not None:
            self._log.logs["test_loss"].append(loss)
        self._visualizer.update(epoch)

    def finalize(self):
        self._visualizer.finalize(background=True)
