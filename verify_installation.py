"""
Installation Verification Script
=================================

Run this script to verify that all dependencies are correctly installed
and the WBC Counter is ready to use.

Usage:
    python verify_installation.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python {version.major}.{version.minor}.{version.micro} "
              f"(requires 3.8+)")
        return False


def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies...")

    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'PyQt6': 'PyQt6',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'grad_cam': 'GradCAM',
        'ttach': 'ttach',
    }

    optional_dependencies = {
        'stitching': 'Image Stitching',
        'bbaug': 'BBox Augmentation',
    }

    all_ok = True

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - not installed")
            all_ok = False

    print("\nOptional dependencies:")
    for module, name in optional_dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ⚠ {name} - not installed (optional)")

    return all_ok


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available")
            print(f"    Devices: {device_count}")
            print(f"    GPU: {device_name}")
            return True
        else:
            print("  ⚠ CUDA not available (CPU-only mode)")
            return False
    except ImportError:
        print("  ✗ Cannot check CUDA (PyTorch not installed)")
        return False


def check_yolov5():
    """Check if YOLOv5 repository is present."""
    print("\nChecking YOLOv5...")
    yolo_path = Path('yolov5')

    if yolo_path.exists() and yolo_path.is_dir():
        print(f"  ✓ YOLOv5 directory found at {yolo_path.absolute()}")
        return True
    else:
        print(f"  ✗ YOLOv5 directory not found")
        print(f"    Please clone it with:")
        print(f"    git clone https://github.com/ultralytics/yolov5")
        return False


def check_weights():
    """Check if model weights are available."""
    print("\nChecking model weights...")

    weights_locations = [
        Path('weights'),
        Path('torch_rcnn_try'),
        Path('.'),
    ]

    found_weights = []
    for location in weights_locations:
        if location.exists():
            pt_files = list(location.glob('*.pt'))
            found_weights.extend(pt_files)

    if found_weights:
        print(f"  ✓ Found {len(found_weights)} weight file(s):")
        for weight in found_weights[:5]:  # Show first 5
            print(f"    - {weight}")
        if len(found_weights) > 5:
            print(f"    ... and {len(found_weights) - 5} more")
        return True
    else:
        print(f"  ✗ No model weights found")
        print(f"    Download from: https://drive.google.com/file/d/1M2y5-pq6S2rZP6y00y5KQ5wn27J2RMgr/view")
        return False


def check_project_structure():
    """Verify project structure."""
    print("\nChecking project structure...")

    required_dirs = [
        'app',
        'torch_rcnn_try',
        'image_stitching',
        'examples',
    ]

    required_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        'LICENSE',
        'CONTRIBUTING.md',
    ]

    all_ok = True

    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - missing")
            all_ok = False

    for file_name in required_files:
        if Path(file_name).exists():
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} - missing")
            all_ok = False

    return all_ok


def print_summary(results):
    """Print summary of verification."""
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)

    all_critical_ok = all([
        results['python'],
        results['dependencies'],
        results['structure'],
    ])

    if all_critical_ok:
        print("✓ All critical components verified successfully!")

        if not results['yolov5']:
            print("\n⚠ YOLOv5 not found - clone it to use the application")
        if not results['weights']:
            print("⚠ Model weights not found - download to run inference")
        if not results['cuda']:
            print("⚠ CUDA not available - will use CPU (slower)")

        print("\nYou're ready to use WBC Counter!")
        print("\nNext steps:")
        if not results['yolov5']:
            print("  1. Clone YOLOv5: git clone https://github.com/ultralytics/yolov5")
        if not results['weights']:
            print("  2. Download model weights from Google Drive")
        print("  3. Run: python app/gui_v2.py --yolo <path> --resnet <path>")

    else:
        print("✗ Some critical components are missing!")
        print("\nPlease fix the issues above and run verification again.")
        print("\nFor help, see: README.md or CONTRIBUTING.md")

    print("="*60 + "\n")


def main():
    """Run all verification checks."""
    print("WBC Counter Installation Verification")
    print("="*60)

    results = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'cuda': check_cuda(),
        'yolov5': check_yolov5(),
        'weights': check_weights(),
        'structure': check_project_structure(),
    }

    print_summary(results)

    # Return exit code
    critical_ok = all([
        results['python'],
        results['dependencies'],
        results['structure'],
    ])
    return 0 if critical_ok else 1


if __name__ == '__main__':
    sys.exit(main())
