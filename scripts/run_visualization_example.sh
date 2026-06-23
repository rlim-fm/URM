#!/bin/bash
# Example: Training with Visualizations
# This script demonstrates how to run URM training with streaming visualizations

set -e

echo "==========================================="
echo "URM Training with Visualizations"
echo "==========================================="

# Check for ffmpeg (required for MP4 generation)
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ Warning: ffmpeg not found"
    echo "   MP4 video generation will not work"
    echo "   Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import matplotlib, scipy, sklearn, h5py" 2>/dev/null || {
    echo "Missing required packages. Installing..."
    pip install -r requirements.txt
}

# Create output directories
mkdir -p visualizations
mkdir -p train_out

# Run the example training script
echo ""
echo "Starting training with visualizations..."
echo "(This is a small example. Modify example_urm_visualization.py for larger runs)"
echo ""

python3 example_urm_visualization.py

echo ""
echo "==========================================="
echo "✓ Training complete!"
echo "Check visualizations/ directory for outputs"
echo "==========================================="
