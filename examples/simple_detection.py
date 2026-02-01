"""
Simple Detection Example
========================

This script demonstrates basic usage of the WBC Counter for detecting
and classifying white blood cells from a microscope image.

Usage:
    python examples/simple_detection.py --image path/to/image.jpg \
                                        --yolo weights/yolo.pt \
                                        --resnet weights/resnet.pt
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from app.two_stage_detector import TwoStageDetector


def load_image(image_path: str):
    """Load and preprocess an image for detection.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (original PIL image, preprocessed tensor)
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor_image = transform(image)

    return image, tensor_image


def visualize_results(image, results, save_path=None):
    """Visualize detection results with bounding boxes.

    Args:
        image: Original PIL image
        results: Detection results dictionary
        save_path: Optional path to save visualization
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Color map for different cell types
    colors = {
        'LY': 'blue', 'RBC': 'red', 'PLT': 'yellow',
        'EO': 'green', 'MO': 'cyan', 'BNE': 'magenta',
        'SEN': 'orange', 'BA': 'pink'
    }

    for result in results:
        for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=colors.get(label, 'white'),
                facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            ax.text(
                x1, y1 - 5,
                f'{label}: {score:.2f}',
                bbox=dict(facecolor=colors.get(label, 'white'), alpha=0.7),
                fontsize=8, color='black'
            )

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def print_summary(results):
    """Print a summary of cell counts.

    Args:
        results: Detection results dictionary
    """
    print("\n" + "="*50)
    print("Cell Detection Summary")
    print("="*50)

    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        print(f"  Total cells detected: {len(result['boxes'])}")

        # Count by cell type
        cell_counts = {}
        for label in result['labels']:
            cell_counts[label] = cell_counts.get(label, 0) + 1

        print("\n  Breakdown by cell type:")
        for cell_type, count in sorted(cell_counts.items()):
            print(f"    {cell_type}: {count}")

        # Filter for WBCs only (exclude RBC and PLT)
        wbc_count = sum(1 for label in result['labels']
                       if label not in ['RBC', 'PLT'])
        print(f"\n  Total WBCs: {wbc_count}")

    print("\n" + "="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Detect and classify white blood cells in microscope images'
    )
    parser.add_argument(
        '--image', type=str, required=True,
        help='Path to microscope image'
    )
    parser.add_argument(
        '--yolo', type=str, required=True,
        help='Path to YOLOv5 model weights'
    )
    parser.add_argument(
        '--resnet', type=str, required=True,
        help='Path to ResNet50 classifier weights'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save visualization (default: display only)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed detection information'
    )

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return

    # Initialize detector
    print("Loading models...")
    detector = TwoStageDetector(
        yolo_path=args.yolo,
        resnet_path=args.resnet,
        model_name='yolo'
    )
    print(f"Using device: {detector.device}")

    # Load image
    print(f"Loading image: {args.image}")
    image, tensor_image = load_image(args.image)

    # Run detection
    print("Running detection...")
    results = detector.detect(tensor_image, verbose=args.verbose)

    # Print summary
    print_summary(results)

    # Visualize results
    print("Generating visualization...")
    visualize_results(image, results, save_path=args.output)


if __name__ == '__main__':
    main()
