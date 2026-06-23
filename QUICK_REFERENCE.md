# URM Visualizations - Quick Reference

## 🚀 Quick Start (30 seconds)

```bash
cd /Users/rlim/Labs/URM

# 1. Check dependencies (optional)
python3 check_dependencies.py

# 2. Run visualization example
python3 example_urm_visualization.py

# 3. Check outputs
ls visualizations/topk-sum/
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| **visualization.py** | Core system (don't modify) |
| **train.py** | Processor with visualization support |
| **viz_utils.py** | Utility functions for metrics collection |
| **VISUALIZATION_GUIDE.md** | Complete documentation |
| **INTEGRATION_SUMMARY.md** | Integration overview |
| **example_urm_visualization.py** | Runnable example |

## 🎯 Common Tasks

### Run Example Training with Visualizations
```bash
python3 example_urm_visualization.py
```

### Use in Your Code
```python
from visualization import Visualizer
from train import Processor

visualizer = Visualizer(name='MyModel')
visualizer.register_loss_history()
visualizer.register_convergence_1d(axis=0)

processor = Processor(..., visualizer=visualizer)
processor.run()  # Visualizations generated automatically!
```

### Create Custom Visualization
```python
from visualization import Visualization

class MyViz(Visualization):
    def update(self, processor, epoch):
        self.data.append(...)
    
    def finalize(self, output_dir, prefix):
        # Generate output files
        pass

visualizer.register(MyViz())
```

### Track Metrics (for pretrain.py)
```python
from viz_utils import MetricsCollector

collector = MetricsCollector(checkpoint_path, run_name)
collector.log_losses(train_loss, test_loss, step)
collector.finalize()
```

### Generate Loss Plot from Saved Metrics
```python
from viz_utils import plot_loss_history

plot_loss_history('checkpoints/my_run/metrics.json')
```

## 📊 Available Visualizations

| Method | Output | What It Does |
|--------|--------|---|
| `register_loss_history()` | PNG | Loss curves |
| `register_convergence_1d()` | MP4 | Function convergence animation |
| `register_pca_3d()` | MP4 | Hidden states in 3D (fixed basis) |
| `register_pca_3d_procrustes()` | MP4 | Hidden states in 3D (smooth) |
| `register(CustomViz())` | Custom | Your custom visualization |

## 🛠️ Shell Scripts

```bash
# Run visualization example
bash scripts/run_visualization_example.sh

# Existing scripts still work
bash scripts/URM_sudoku.sh
bash scripts/URM_arcagi1.sh
```

## ✅ Requirements

- ✓ Python 3.7+
- ✓ torch 2.8.0
- ✓ numpy, scipy, scikit-learn, matplotlib, h5py
- ✓ ffmpeg (for MP4 generation)

Install:
```bash
pip install -r requirements.txt
brew install ffmpeg  # macOS
```

## 📍 Output Locations

| Output | Location |
|--------|----------|
| Visualizations | `visualizations/topk-sum/` |
| Training data | `train_out/` |
| Metrics | `checkpoints/{run_name}/metrics.json` |

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| ffmpeg not found | `brew install ffmpeg` |
| Import error | Run `check_dependencies.py` |
| Out of memory | Use `Visualizer(sampling=10)` |
| No visualizations | Check that visualizer is passed to Processor |

## 📚 Full Documentation

- **VISUALIZATION_GUIDE.md** - Complete guide with examples
- **INTEGRATION_SUMMARY.md** - What was integrated and how
- **README_BrainSpace.md** - Original BrainSpace documentation

## 🎬 Example Output Files

After running `example_urm_visualization.py`:

```
visualizations/topk-sum/
├── URM-Example_loss_history.png           # Static plot
├── URM-Example_1d_convergence.mp4         # MP4 animation
├── URM-Example_pca_3d_anchor.mp4          # 3D animation
└── URM-Example_pca_3d_procrustes.mp4      # 3D animation (smooth)
```

## 💡 Tips

1. **Memory**: Use `Visualizer(sampling=5)` to sample every 5th epoch
2. **Speed**: Video generation takes ~1-2 seconds per animation
3. **Quality**: 150 DPI PNG, 15 FPS MP4 (configurable)
4. **Custom**: Easy to extend - inherit from `Visualization` class

## 🎓 Learn More

```python
# See how it works
from visualization import Visualizer, Visualization

# Read the docstrings
help(Visualizer)
help(Visualization)
help(Visualizer.register_loss_history)
```

---

**Quick Links:**
- 🏠 Main Repo: /Users/rlim/Labs/URM
- 📖 Docs: See VISUALIZATION_GUIDE.md and INTEGRATION_SUMMARY.md
- 🧪 Test: `python3 example_urm_visualization.py`
- 🐛 Check: `python3 check_dependencies.py`
