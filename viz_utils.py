"""
Visualization utilities for URM training.

This module provides helpers to integrate the BrainSpace visualization system
with the pretrain.py training loop.

Usage:
    # During training, collect metrics
    from viz_utils import MetricsCollector
    collector = MetricsCollector(checkpoint_path)

    # After each evaluation step, log metrics
    collector.log_losses(train_loss, test_loss, step)

    # After training, generate visualizations
    collector.finalize()
"""

import os
import json
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path


class MetricsCollector:
    """Collects training metrics for visualization."""

    def __init__(self, checkpoint_path: Optional[str] = None, name: str = 'URM'):
        """
        Initialize metrics collector.

        Args:
            checkpoint_path: Path to save metrics and visualizations
            name: Model/experiment name for visualization output
        """
        self.checkpoint_path = checkpoint_path or 'checkpoints/default'
        self.name = name
        self.metrics = {
            'train_loss': [],
            'test_loss': [],
            'steps': []
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)

    def log_losses(self, train_loss: float, test_loss: float, step: int):
        """Log training metrics at a specific step."""
        self.metrics['train_loss'].append(float(train_loss))
        self.metrics['test_loss'].append(float(test_loss))
        self.metrics['steps'].append(int(step))

    def save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = os.path.join(self.checkpoint_path, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {metrics_file}")

    def finalize(self):
        """Finalize and save metrics."""
        self.save_metrics()
        self._generate_summary()

    def _generate_summary(self):
        """Generate a summary of training metrics."""
        if not self.metrics['train_loss']:
            print("No metrics collected.")
            return

        train_losses = np.array(self.metrics['train_loss'])
        test_losses = np.array(self.metrics['test_loss'])

        summary = {
            'final_train_loss': float(train_losses[-1]),
            'final_test_loss': float(test_losses[-1]),
            'best_train_loss': float(train_losses.min()),
            'best_test_loss': float(test_losses.min()),
            'best_train_step': int(self.metrics['steps'][np.argmin(train_losses)]),
            'best_test_step': int(self.metrics['steps'][np.argmin(test_losses)]),
        }

        summary_file = os.path.join(self.checkpoint_path, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print("="*60 + "\n")


def plot_loss_history(metrics_file: str, output_dir: Optional[str] = None):
    """
    Generate a loss history plot from metrics JSON file.

    Args:
        metrics_file: Path to metrics.json file
        output_dir: Directory to save plot (defaults to same as metrics file)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    output_dir = output_dir or os.path.dirname(metrics_file)
    os.makedirs(output_dir, exist_ok=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = metrics['steps']
    train_losses = metrics['train_loss']
    test_losses = metrics['test_loss']

    ax.plot(steps, train_losses, 'b-', linewidth=2, label='Train Loss', marker='o', markersize=4)
    ax.plot(steps, test_losses, 'r-', linewidth=2, label='Test Loss', marker='s', markersize=4)

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss History', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()

    output_file = os.path.join(output_dir, 'loss_history.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Loss history plot saved to {output_file}")


if __name__ == '__main__':
    # Example usage
    collector = MetricsCollector(checkpoint_path='checkpoints/test_viz')
    collector.log_losses(0.5, 0.48, 0)
    collector.log_losses(0.3, 0.35, 1000)
    collector.log_losses(0.15, 0.25, 2000)
    collector.finalize()
