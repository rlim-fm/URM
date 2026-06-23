"""
Example: Training URM with streaming visualizations.

This example shows how to use the visualization system imported from BrainSpace
with the URM architecture. It demonstrates:

1. Creating a visualizer with desired visualizations
2. Passing it to the training loop
3. Generating MP4 animations after training

To run:
    python example_urm_visualization.py

Requirements:
    - ffmpeg must be installed: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Try to import URM models (these would be actual URM models)
try:
    from models import SimpleTransformerModel
    from train import Processor
    from visualization import Visualizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required models and dependencies installed.")
    exit(1)


def main():
    """Train a simple model with visualizations."""

    print("=" * 70)
    print("URM TRAINING WITH STREAMING VISUALIZATIONS")
    print("=" * 70)

    # Create visualizer with desired visualizations
    visualizer = Visualizer(name='URM-Example')

    # Register visualizations you want to generate
    visualizer.register_loss_history()           # Loss plot
    visualizer.register_convergence_1d(axis=0)   # 1D convergence animation

    # Note: PCA visualizations require specific data dimensions
    # visualizer.register_pca_3d()                 # 3D PCA (anchor mode)
    # visualizer.register_pca_3d_procrustes()      # 3D PCA (Procrustes mode)

    print("\n[Visualizations Registered]")
    for name in visualizer.visualizations.keys():
        print(f"  ✓ {name}")

    # Create model and optimizer
    print("\n[Building Model]")
    model = SimpleTransformerModel(input_dim=1, dropout=0.25, tropical=True)
    print(f"  Model: {model.__class__.__name__}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.75)
    print(f"  Optimizer: AdamW")
    print(f"  Scheduler: StepLR")

    # Create processor with visualizer
    print("\n[Creating Training Processor]")
    processor = Processor(
        x_range=(-8, 8),
        data_dim=(10, 1),
        N=2048,
        ground_truth=None,  # Will use default topksubset(3, 1)
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=100,  # For demo; use 5000 for real training
        criterion=nn.MSELoss(reduction='mean'),
        visualizer=visualizer,
        seed=42,
        device='cpu'  # Change to 'cuda' if available
    )

    print(f"  Training Epochs: {processor.epochs}")
    print(f"  Training Samples: {len(processor.x_train)}")
    print(f"  Test Samples: {len(processor.x_test)}")

    # Run training
    print("\n[Starting Training...]")
    print("  Training with streaming visualizations (memory efficient!)")
    print("  Visualizations will be saved after training completes.\n")

    processor.run()

    # Print summary
    print("\n")
    processor.print_summary()

    # Save model and data
    output_dir = Path('train_out')
    output_dir.mkdir(exist_ok=True)

    processor.save(
        filename=str(output_dir / 'urm_training_data.h5'),
        output_dir=str(output_dir)
    )

    print("\n[Visualization Output]")
    print("  Animations and plots saved to: visualizations/topk-sum/")
    print("  Expected files:")
    print("    - URM-Example_loss_history.png")
    print("    - URM-Example_1d_convergence.mp4")

    print("\n✓ Training and visualization complete!")
    print("="*70)


if __name__ == '__main__':
    main()
