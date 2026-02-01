# Contributing to WBC Counter

Thank you for your interest in contributing to the White Blood Cell Counter project! This document provides guidelines and instructions for contributing.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, GPU info)
- Screenshots or error messages if applicable

### Suggesting Enhancements

We welcome enhancement suggestions! Please create an issue with:

- A clear description of the enhancement
- Use cases and benefits
- Implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards (see below)

3. **Test your changes** thoroughly:
   ```bash
   # Run tests (if available)
   pytest tests/

   # Test the GUI
   python app/gui_v2.py --yolo weights/yolo.pt --resnet weights/resnet.pt
   ```

4. **Commit your changes** with clear, descriptive messages:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```

5. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## 💻 Development Setup

### Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .[dev]  # Install in editable mode with dev dependencies
   ```

3. Clone YOLOv5:
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5 && pip install -r requirements.txt && cd ..
   ```

## 📝 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Grouped and sorted (stdlib, third-party, local)
- **Docstrings**: Google style for all public functions and classes
- **Type hints**: Use where it improves clarity

### Code Formatting

We use **Black** for code formatting:

```bash
# Format all Python files
black app/ torch_rcnn_try/ image_stitching/

# Check formatting without changes
black --check app/
```

### Linting

Run **flake8** to catch common issues:

```bash
flake8 app/ --max-line-length=100 --ignore=E203,W503
```

### Documentation

- Add docstrings to all public functions and classes
- Update README.md if you add new features
- Add inline comments for complex logic

Example docstring:

```python
def detect_cells(image_path: str, threshold: float = 0.5) -> dict:
    """Detect and classify cells in a microscope image.

    Args:
        image_path: Path to the microscope image file.
        threshold: Confidence threshold for detection (0.0-1.0).

    Returns:
        Dictionary containing:
            - boxes: List of bounding box coordinates [x1, y1, x2, y2]
            - labels: List of cell type labels
            - scores: List of confidence scores

    Raises:
        FileNotFoundError: If image_path does not exist.
        ValueError: If threshold is not in range [0, 1].
    """
    # Implementation here
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov=torch_rcnn_try

# Run specific test file
pytest tests/test_detector.py
```

### Writing Tests

Add tests for new features in the `tests/` directory:

```python
import pytest
from app.two_stage_detector import TwoStageDetector

def test_detector_initialization():
    """Test that detector initializes correctly."""
    detector = TwoStageDetector(
        yolo_path='weights/yolo.pt',
        resnet_path='weights/resnet.pt'
    )
    assert detector.device is not None
    assert detector.model is not None
```

## 🏗️ Project Architecture

### Key Components

1. **app/two_stage_detector.py**: Core detection logic
2. **app/gui_v2.py**: PyQt6 GUI application
3. **app/image_splitting.py**: Image preprocessing and utilities
4. **app/workers.py**: Async worker threads for GUI
5. **torch_rcnn_try/**: Training scripts and experiments

### Adding New Features

When adding a new feature:

1. **Plan the architecture**: Consider how it integrates with existing code
2. **Update docstrings**: Document new functions and classes
3. **Add tests**: Ensure your feature works correctly
4. **Update README**: Add usage examples if it's user-facing

## 🎓 Training Models

### Dataset Preparation

1. **Image format**: JPG or PNG, any resolution
2. **Annotation format**: CSV with columns:
   ```
   image_path,xmin,ymin,xmax,ymax,class
   data/img1.jpg,100,200,150,250,0
   ```

3. **Class mapping**: Use standard indices:
   - 0: Lymphocyte (LY)
   - 1: Red Blood Cell (RBC)
   - 2: Platelet (PLT)
   - 3: Eosinophil (EO)
   - 4: Monocyte (MO)
   - 5: Band Neutrophil (BNE)
   - 6: Segmented Neutrophil (SEN)
   - 7: Basophil (BA)

### Training YOLOv5 Detector

```bash
cd yolov5

# Create data.yaml
cat > data.yaml << EOF
train: ../datasets/train/images
val: ../datasets/val/images
nc: 8  # number of classes
names: ['LY', 'RBC', 'PLT', 'EO', 'MO', 'BNE', 'SEN', 'BA']
EOF

# Train
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 100 \
    --data data.yaml \
    --weights yolov5m.pt \
    --project ../runs \
    --name yolo_wbc
```

### Training ResNet50 Classifier

```bash
cd torch_rcnn_try

python torch_resnet50_balanced.py \
    --data_path ../datasets/centered_cells/ \
    --labels_csv ../datasets/labels_balanced.csv \
    --epochs 80 \
    --batch_size 32 \
    --lr 0.001 \
    --save_dir ../runs/resnet50_wbc
```

### Hyperparameter Tuning

Key hyperparameters to tune:

**YOLOv5:**
- `--img`: Input image size (640, 1280)
- `--batch`: Batch size (16, 32)
- `--epochs`: Training epochs (100-200)
- `--hyp`: Hyperparameter config file

**ResNet50:**
- `--lr`: Learning rate (0.001, 0.0001)
- `--batch_size`: Batch size (16, 32, 64)
- `--epochs`: Training epochs (50-100)
- `--augmentation`: Enable/disable data augmentation

## 📊 Model Evaluation

### Evaluation Metrics

Use the evaluation notebooks:

```bash
jupyter notebook app/evaluate_detector.ipynb
```

Key metrics to report:
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **Per-class Precision/Recall**: For each cell type
- **Inference Speed**: FPS on your hardware
- **Confusion Matrix**: Misclassification analysis

## 🐛 Debugging Tips

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size
   - Use smaller image size
   - Enable gradient accumulation

2. **YOLOv5 not found**:
   - Ensure yolov5 directory is at project root
   - Check `yolo_repo_path` parameter

3. **PyQt6 GUI not showing**:
   - Check display environment variables
   - Try running with `QT_DEBUG_PLUGINS=1`

### Logging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your code
detector.detect(image, verbose=True)
```

## 🔬 Research Contributions

If your contribution relates to research:

1. **Document methodology**: Explain the approach in detail
2. **Provide benchmarks**: Compare with baseline methods
3. **Share datasets**: If possible, share data or provide access
4. **Write documentation**: Add to docs or create notebooks

## 📜 Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Help others learn and grow
- Focus on the work, not the person
- Keep discussions on-topic

## ❓ Questions?

If you have questions:

- Check the [README](README.md) first
- Search existing issues
- Create a new issue with the "question" label
- Reach out to the maintainers

## 🎉 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Academic publications (if applicable)

Thank you for contributing to WBC Counter!
