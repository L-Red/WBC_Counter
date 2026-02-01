# WBC Counter Examples

This directory contains example scripts demonstrating various use cases of the WBC Counter.

## 📁 Available Examples

### 1. Simple Detection (`simple_detection.py`)

Detect and classify cells in a single image with visualization.

**Usage:**
```bash
python examples/simple_detection.py \
    --image path/to/microscope_image.jpg \
    --yolo weights/yolo_best.pt \
    --resnet weights/resnet50_classifier.pt \
    --output results.png
```

**Features:**
- Single image processing
- Visual bounding box output
- Cell count summary
- Color-coded cell types

**Example Output:**
```
Cell Detection Summary
==================================================
Image 1:
  Total cells detected: 47

  Breakdown by cell type:
    BNE: 3
    EO: 2
    LY: 15
    MO: 4
    RBC: 18
    SEN: 5

  Total WBCs: 29
==================================================
```

---

### 2. Batch Processing (`batch_processing.py`)

Process multiple images and generate CSV reports.

**Usage:**
```bash
python examples/batch_processing.py \
    --input_dir path/to/images/ \
    --yolo weights/yolo_best.pt \
    --resnet weights/resnet50_classifier.pt \
    --output results.csv
```

**Features:**
- Batch processing with progress bar
- CSV output with per-image counts
- Summary statistics
- Error handling for corrupted images

**Example CSV Output:**
```csv
filename,total_cells,wbc_count,BA,BNE,EO,LY,MO,PLT,RBC,SEN
image_001.jpg,45,28,0,3,2,12,4,5,10,7
image_002.jpg,52,31,1,4,1,15,3,8,13,5
```

---

## 🚀 Quick Start

1. **Download model weights** from [Google Drive](https://drive.google.com/file/d/1M2y5-pq6S2rZP6y00y5KQ5wn27J2RMgr/view?usp=sharing)

2. **Organize your directory:**
   ```
   WBC_Counter/
   ├── weights/
   │   ├── yolo_best.pt
   │   └── resnet50_classifier.pt
   ├── examples/
   └── test_images/
       ├── sample1.jpg
       └── sample2.jpg
   ```

3. **Run an example:**
   ```bash
   # Single image
   python examples/simple_detection.py \
       --image test_images/sample1.jpg \
       --yolo weights/yolo_best.pt \
       --resnet weights/resnet50_classifier.pt

   # Batch processing
   python examples/batch_processing.py \
       --input_dir test_images/ \
       --yolo weights/yolo_best.pt \
       --resnet weights/resnet50_classifier.pt
   ```

---

## 🎨 Customization

### Modifying Visualization Colors

Edit the `colors` dictionary in `simple_detection.py`:

```python
colors = {
    'LY': 'blue',      # Lymphocytes
    'RBC': 'red',      # Red Blood Cells
    'PLT': 'yellow',   # Platelets
    'EO': 'green',     # Eosinophils
    'MO': 'cyan',      # Monocytes
    'BNE': 'magenta',  # Band Neutrophils
    'SEN': 'orange',   # Segmented Neutrophils
    'BA': 'pink'       # Basophils
}
```

### Filtering Cell Types

To only count specific cell types, modify the WBC count calculation:

```python
# Only count neutrophils and lymphocytes
wbc_count = sum(1 for label in result['labels']
               if label in ['SEN', 'BNE', 'LY'])
```

### Adjusting Confidence Threshold

Modify the detection parameters in `two_stage_detector.py`:

```python
# Filter low-confidence detections
results = detector.detect(
    tensor_image,
    verbose=True,
    confidence_threshold=0.7  # Increase for higher precision
)
```

---

## 📊 Advanced Usage

### Using with Jupyter Notebooks

```python
import sys
sys.path.insert(0, '../')

from app.two_stage_detector import TwoStageDetector
from examples.simple_detection import load_image, visualize_results

# Load detector
detector = TwoStageDetector(
    yolo_path='../weights/yolo_best.pt',
    resnet_path='../weights/resnet50_classifier.pt'
)

# Process image
image, tensor_image = load_image('sample.jpg')
results = detector.detect(tensor_image)

# Visualize
visualize_results(image, results)
```

### Programmatic Access to Results

```python
# Access detailed results
for result in results:
    boxes = result['boxes']           # [[x1, y1, x2, y2], ...]
    labels = result['labels']         # ['LY', 'SEN', ...]
    scores = result['scores']         # [0.95, 0.87, ...]
    yolo_labels = result['yolo_labels']  # YOLO predictions
    yolo_scores = result['yolo_scores']  # YOLO confidences

    # Process each detection
    for box, label, score in zip(boxes, labels, scores):
        print(f"Found {label} at {box} with confidence {score:.2f}")
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory

If you get CUDA OOM errors with batch processing:

```python
# Process images one at a time
# Add torch.cuda.empty_cache() after each image
import torch

# After processing each image:
torch.cuda.empty_cache()
```

### Import Errors

Make sure you're running from the project root or have added it to your path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Slow Processing

For faster processing:
- Use GPU if available
- Reduce image size before detection
- Set `small_image=True` for smaller images

---

## 📚 Additional Resources

- [Main README](../README.md) - Project overview and setup
- [CONTRIBUTING](../CONTRIBUTING.md) - Development guidelines
- [API Documentation](../docs/) - Detailed API reference

---

## 💡 Tips

1. **Use consistent image quality**: Better results with high-resolution, well-focused images
2. **Proper illumination**: Ensure uniform lighting across the microscope field
3. **Calibration**: Validate counts on known samples before production use
4. **Batch size**: Process 10-50 images at a time for optimal performance

---

**Questions?** Open an issue on [GitHub](https://github.com/L-Red/WBC_Counter/issues)
