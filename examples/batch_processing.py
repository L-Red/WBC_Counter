"""
Batch Processing Example
=========================

Process multiple images and generate a CSV report with cell counts.

Usage:
    python examples/batch_processing.py --input_dir path/to/images/ \
                                        --yolo weights/yolo.pt \
                                        --resnet weights/resnet.pt \
                                        --output results.csv
"""

import argparse
from pathlib import Path
import sys
import csv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from app.two_stage_detector import TwoStageDetector


def process_directory(input_dir, detector, verbose=False):
    """Process all images in a directory.

    Args:
        input_dir: Path to directory containing images
        detector: TwoStageDetector instance
        verbose: Whether to print detailed information

    Returns:
        List of dictionaries containing results for each image
    """
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}

    # Find all images
    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {input_dir}")
        return []

    print(f"Found {len(image_files)} images to process")

    results = []
    transform = transforms.ToTensor()

    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            # Load and process image
            image = Image.open(image_file).convert('RGB')
            tensor_image = transform(image)

            # Run detection
            detection_results = detector.detect(
                tensor_image,
                verbose=verbose
            )

            # Count cells by type
            cell_counts = {}
            for result in detection_results:
                for label in result['labels']:
                    cell_counts[label] = cell_counts.get(label, 0) + 1

            # Calculate total WBCs (exclude RBC and PLT)
            wbc_count = sum(
                count for label, count in cell_counts.items()
                if label not in ['RBC', 'PLT']
            )

            results.append({
                'filename': image_file.name,
                'total_cells': sum(len(r['boxes']) for r in detection_results),
                'wbc_count': wbc_count,
                'cell_counts': cell_counts
            })

        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
            continue

    return results


def save_results_csv(results, output_path):
    """Save results to CSV file.

    Args:
        results: List of result dictionaries
        output_path: Path to save CSV file
    """
    if not results:
        print("No results to save")
        return

    # Get all unique cell types across all images
    all_cell_types = set()
    for result in results:
        all_cell_types.update(result['cell_counts'].keys())
    all_cell_types = sorted(all_cell_types)

    # Write CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'total_cells', 'wbc_count'] + all_cell_types
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            row = {
                'filename': result['filename'],
                'total_cells': result['total_cells'],
                'wbc_count': result['wbc_count']
            }
            # Add individual cell type counts
            for cell_type in all_cell_types:
                row[cell_type] = result['cell_counts'].get(cell_type, 0)

            writer.writerow(row)

    print(f"\nResults saved to {output_path}")


def print_summary_stats(results):
    """Print summary statistics across all images.

    Args:
        results: List of result dictionaries
    """
    if not results:
        return

    total_images = len(results)
    total_cells = sum(r['total_cells'] for r in results)
    total_wbcs = sum(r['wbc_count'] for r in results)

    # Aggregate cell counts
    aggregate_counts = {}
    for result in results:
        for cell_type, count in result['cell_counts'].items():
            aggregate_counts[cell_type] = aggregate_counts.get(cell_type, 0) + count

    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Total cells detected: {total_cells}")
    print(f"Total WBCs detected: {total_wbcs}")
    print(f"Average cells per image: {total_cells/total_images:.1f}")
    print(f"Average WBCs per image: {total_wbcs/total_images:.1f}")

    print("\nAggregate cell counts:")
    for cell_type, count in sorted(aggregate_counts.items()):
        percentage = (count / total_cells * 100) if total_cells > 0 else 0
        print(f"  {cell_type}: {count} ({percentage:.1f}%)")

    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Batch process microscope images for WBC counting'
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Directory containing microscope images'
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
        '--output', type=str, default='results.csv',
        help='Path to save CSV results (default: results.csv)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed detection information'
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not Path(args.input_dir).exists():
        print(f"Error: Directory not found at {args.input_dir}")
        return

    # Initialize detector
    print("Loading models...")
    detector = TwoStageDetector(
        yolo_path=args.yolo,
        resnet_path=args.resnet,
        model_name='yolo'
    )
    print(f"Using device: {detector.device}")

    # Process all images
    results = process_directory(args.input_dir, detector, args.verbose)

    if results:
        # Save results
        save_results_csv(results, args.output)

        # Print summary
        print_summary_stats(results)
    else:
        print("No images were successfully processed")


if __name__ == '__main__':
    main()
