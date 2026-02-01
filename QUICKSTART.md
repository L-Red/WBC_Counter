# Quick Start Guide

Get up and running with WBC Counter in 5 minutes!

## ⚡ Installation (5 minutes)

### Step 1: Clone the Repository
```bash
git clone https://github.com/L-Red/WBC_Counter.git
cd WBC_Counter
```

### Step 2: Set Up Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install main dependencies
pip install -r requirements.txt

# Clone and install YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

### Step 4: Verify Installation
```bash
python verify_installation.py
```

You should see:
```
✓ All critical components verified successfully!
```

## 🎯 Get Model Weights

1. Download weights from [Google Drive](https://drive.google.com/file/d/1M2y5-pq6S2rZP6y00y5KQ5wn27J2RMgr/view?usp=sharing)

2. Create a weights directory and extract:
```bash
mkdir weights
# Extract downloaded weights here
```

You should have:
- `weights/yolo_best.pt`
- `weights/resnet50_classifier.pt`

## 🚀 Run Your First Detection

### Option 1: GUI Application
```bash
python app/gui_v2.py --yolo weights/yolo_best.pt --resnet weights/resnet50_classifier.pt
```

Then:
1. Click "Open Image"
2. Select a microscope image
3. Click "Count Cells"
4. View results!

### Option 2: Simple Script
```bash
python examples/simple_detection.py \
    --image path/to/your/image.jpg \
    --yolo weights/yolo_best.pt \
    --resnet weights/resnet50_classifier.pt \
    --output results.png
```

### Option 3: Programmatic Usage

Create `test.py`:
```python
from app.two_stage_detector import TwoStageDetector
from torchvision import transforms
from PIL import Image

# Load detector
detector = TwoStageDetector(
    yolo_path='weights/yolo_best.pt',
    resnet_path='weights/resnet50_classifier.pt'
)

# Load image
image = Image.open('microscope_image.jpg')
tensor_image = transforms.ToTensor()(image)

# Detect cells
results = detector.detect(tensor_image)

# Print counts
for result in results:
    print(f"Detected {len(result['boxes'])} cells")
    for label in result['labels']:
        print(f"  - {label}")
```

Run it:
```bash
python test.py
```

## 🎓 Next Steps

- **Batch Processing**: See `examples/batch_processing.py`
- **Training**: Read [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Reference**: Check [README.md](README.md)
- **Issues?** See troubleshooting below

## 🐛 Common Issues

### "No module named PyQt6"
```bash
pip install PyQt6
```

### "CUDA out of memory"
The models will automatically use CPU if CUDA is unavailable. For better performance with limited GPU memory:
- Process smaller images
- Close other GPU applications

### "YOLOv5 not found"
Make sure you cloned YOLOv5 in the project root:
```bash
git clone https://github.com/ultralytics/yolov5
```

### "Model weights not found"
Download from Google Drive and place in `weights/` directory.

## 💡 Tips

1. **Use GPU**: ~10x faster than CPU
2. **Good images**: Well-focused, uniform lighting
3. **Start small**: Test on 1-2 images first
4. **Check results**: Verify accuracy on known samples

## 📚 Learn More

- Full documentation: [README.md](README.md)
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Examples: [examples/README.md](examples/README.md)
- Issues: [GitHub Issues](https://github.com/L-Red/WBC_Counter/issues)

---

**Ready to go?** Start with the GUI application and explore from there! 🚀
