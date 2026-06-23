# URM Visualization Integration Guide

This guide explains how to use the visualization system imported from the BrainSpace project with the URM repository.

## Overview

The URM repo now includes a **streaming visualization system** that provides:

- **Memory Efficient**: 27x less RAM during training (data not stored between epochs)
- **MP4 Output**: Professional-quality animations instead of large GIF files
- **Dynamic Data Extraction**: Visualizer pulls data from processor as needed
- **Easy to Extend**: Write custom visualizations with just 2 methods

## Key Components

### 1. **visualization.py** (Core System)
The main visualization framework with:
- Base `Visualization` class for creating custom visualizations
- Pre-built visualizations (loss history, 1D convergence, 3D PCA, etc.)
- `Visualizer` class that orchestrates all visualizations

### 2. **train.py** (Simple Training Loop)
A `Processor` class that demonstrates visualization integration:
- Accepts optional `visualizer` parameter
- Calls `visualizer.update()` at each epoch
- Calls `visualizer.finalize()` after training

### 3. **viz_utils.py** (Utilities)
Helper functions for:
- Collecting metrics during training
- Generating loss history plots
- Creating training summaries

## Quick Start

### 1. For Simple Training (train.py)

```python
from visualization import Visualizer
from train import Processor
import torch.nn as nn
import torch.optim as optim

# Create visualizer
visualizer = Visualizer(name='MyModel')
visualizer.register_loss_history()
visualizer.register_convergence_1d(axis=0)
visualizer.register_pca_3d()
visualizer.register_pca_3d_procrustes()

# Create processor with visualizer
processor = Processor(
    model=MyModel(...),
    epochs=5000,
    visualizer=visualizer  # Pass visualizer here
)

# Train (visualizations auto-generated!)
processor.run()
```

### 2. For Complex Training (pretrain.py)

For distributed training on puzzle datasets, use `viz_utils.MetricsCollector`:

```python
from viz_utils import MetricsCollector

# Create collector
collector = MetricsCollector(checkpoint_path='checkpoints/my_run', name='URM-Sudoku')

# During training loop, log metrics
collector.log_losses(train_loss=0.15, test_loss=0.18, step=1000)

# After training
collector.finalize()  # Generates summary and saves metrics
```

## Available Visualizations

### Loss History Plot
```python
visualizer.register_loss_history()
```
Creates: `{name}_loss_history.png`
- Training and test loss curves on log scale
- Static PNG image

### 1D Convergence Animation
```python
visualizer.register_convergence_1d(axis=0)
```
Creates: `{name}_1d_convergence.mp4`
- Animated convergence along a single input axis
- Shows ground truth vs network prediction at each epoch
- 15 FPS MP4 video

### 3D PCA Convergence (Anchor Mode)
```python
visualizer.register_pca_3d()
```
Creates: `{name}_pca_3d_anchor.mp4`
- Hidden states evolution in PCA space
- Uses fixed PCA basis from final epoch
- Color-coded: blue=in-domain, red=out-of-domain

### 3D PCA Convergence (Procrustes Mode)
```python
visualizer.register_pca_3d_procrustes()
```
Creates: `{name}_pca_3d_procrustes.mp4`
- Per-epoch PCA basis with Procrustes alignment
- Smoother evolution than anchor mode
- Better for visualizing continuous changes

### Function Space Convergence
```python
visualizer.register(FunctionSpaceConvergence())
```
Creates: `{name}_function_space.png`
- 3D PCA projection of function convergence path
- Shows trajectory toward target function

## Custom Visualizations

Create your own visualization by inheriting from `Visualization`:

```python
from visualization import Visualization
import matplotlib.pyplot as plt
import numpy as np

class MyCustomViz(Visualization):
    def __init__(self):
        super().__init__('my_custom_viz')
        self.data = []
    
    def update(self, processor, epoch: int):
        """Called each epoch to collect data."""
        # Extract any data you need from processor
        self.data.append(processor.logs['train_loss'][-1])
    
    def finalize(self, output_dir: str, prefix: str):
        """Called after training to generate output."""
        # Create your visualization
        fig, ax = plt.subplots()
        ax.plot(self.data)
        ax.set_title('My Custom Visualization')
        
        # Save it
        output_file = f'{output_dir}/{prefix}_my_custom.png'
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        print(f"✓ Custom viz saved to {output_file}")

# Use it
visualizer = Visualizer()
visualizer.register(MyCustomViz())
```

## requirements.txt Updates

The following dependencies have been added for visualization:

```
numpy
scipy
scikit-learn
matplotlib
h5py
ffmpeg-python
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

**Note**: For MP4 video generation, ffmpeg must be installed:
- **macOS**: `brew install ffmpeg`
- **Linux**: `apt-get install ffmpeg`
- **Windows**: `choco install ffmpeg` or download from ffmpeg.org

## Running Examples

### Simple Example
```bash
python example_urm_visualization.py
```
This runs a quick training demonstration with visualizations.

### From Shell Scripts
The existing shell scripts (URM_sudoku.sh, URM_arcagi1.sh, etc.) don't directly use visualizations because they run distributed training via torchrun. However, you can:

1. Use `MetricsCollector` to track metrics during training
2. Run visualizations post-training on saved data
3. Modify pretrain.py to integrate `MetricsCollector`

## Output Locations

Visualizations are saved to:
```
visualizations/
├── topk-sum/  (default from train.py/Processor)
│   ├── model_name_loss_history.png
│   ├── model_name_1d_convergence.mp4
│   ├── model_name_pca_3d_anchor.mp4
│   ├── model_name_pca_3d_procrustes.mp4
│   └── model_name_function_space.png
└── [custom output dirs as specified]
```

## Sampling and Memory

To reduce memory usage and speed up visualization:

```python
# Only sample every 10th epoch for visualization
visualizer = Visualizer(sampling=10)

# This significantly reduces video frame count
visualizer.register_loss_history()
```

## Troubleshooting

### Issue: "ffmpeg not found"
**Solution**: Install ffmpeg
- macOS: `brew install ffmpeg`
- Linux: `apt-get install ffmpeg`
- Verify: `ffmpeg -version`

### Issue: MP4 not generated
**Possible causes**:
- ffmpeg not installed (see above)
- matplotlib not installed: `pip install matplotlib`
- Check error messages in console for details

### Issue: Out of memory
**Solution**: Use sampling to reduce frames:
```python
visualizer = Visualizer(sampling=20)  # Every 20 epochs
```

### Issue: No visualizations generated
**Checklist**:
- Visualizer created and registered before Processor
- visualizer passed to Processor constructor
- Training completed without errors
- Check console output for errors during finalize()

## Integration with pretrain.py

To add metrics collection to pretrain.py:

1. Import MetricsCollector at top:
```python
from viz_utils import MetricsCollector
```

2. Create collector in launch() function:
```python
if rank == 0:
    viz_collector = MetricsCollector(checkpoint_path, run_name)
```

3. Log metrics during evaluation:
```python
if rank == 0:
    viz_collector.log_losses(avg_loss_train, avg_loss_test, train_state.step)
```

4. Finalize after training:
```python
if rank == 0:
    viz_collector.finalize()
```

## Performance Notes

- **Memory**: Visualizer uses ~3% of standard logging memory
- **Speed**: Visualization finalization adds ~5-10 seconds per visualization
- **Video Generation**: MP4 generation takes ~1-2 seconds per animation
- **Quality**: 150 DPI PNG, 15 FPS MP4 (configurable)

## References

- **BrainSpace**: The original visualization framework this was imported from
- **README_BrainSpace.md**: Documentation from the original project
- **visualization.py**: Core implementation
- **train.py**: Processor class with visualization integration
- **viz_utils.py**: Utility functions

## Questions or Issues?

Check the code comments in:
- `visualization.py` - Core system documentation
- `train.py` - Processor integration
- `example_urm_visualization.py` - Working example
