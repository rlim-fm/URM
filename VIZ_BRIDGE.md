# Viz Bridge

`viz_bridge.py` connects URM/TARM's puzzle training loop (`pretrain.py`) to
Tropical-RNN's streaming `Visualizer`, producing a loss-history plot alongside
a normal training run (in addition to wandb logging).

## How it works

- `Tropical-RNN/visualization.py` is loaded in isolation via `importlib`
  (not a normal package import), so it never registers top-level modules
  named `models`, `train`, or `datasets` that would collide with URM's own
  `models/` package.
- `PuzzleTrainingLog` is a minimal stand-in for Tropical-RNN's `Processor`,
  satisfying the `.logs` / `.metadata` contract read by
  `Visualization.update(processor, epoch)`.
- `_extract_loss` pulls a scalar loss out of a wandb.log-shaped metrics dict:
  train metrics are flat (`train/lm_loss`), eval metrics are nested
  (`{set_name: {"lm_loss": ..., ...}}`). Either way, the first key ending in
  `lm_loss` is used.
- `PuzzleVisualizerBridge` wraps a `Visualizer` registered with a single
  `LossHistoryPlot`, and exposes:
  - `log_train(metrics, epoch)` — append a train loss point and redraw.
  - `log_eval(metrics, epoch)` — append a test loss point and redraw.
  - `finalize()` — render the final plot (runs in the background).

## Usage

Enable it by passing `+visualize=True` to `pretrain.py`. See
`scripts/URM_sudoku_viz.sh` for a working example (a copy of
`scripts/URM_sudoku.sh` with visualization turned on).

Wiring lives in `pretrain.py`:
```python
if config.visualize:
    from viz_bridge import PuzzleVisualizerBridge
    viz_bridge = PuzzleVisualizerBridge(
        output_dir=os.path.join(config.checkpoint_path or "visualizations", "visualizations"),
        name=config.run_name,
    )
```

Output (the rendered loss-history plot) is written under
`<checkpoint_path>/visualizations/`.

## Extending

`PuzzleVisualizerBridge` currently only registers `LossHistoryPlot`. To
stream additional visualizations, register more `Visualization` subclasses
on `self._visualizer` in `viz_bridge.py`. See
`Tropical-RNN/docs/CUSTOM_VIZ.md` for the `Visualization` base class API
(`update()` / `finalize()`) and examples of writing custom visualizations.
