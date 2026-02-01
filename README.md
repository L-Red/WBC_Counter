# White Blood Cell Counter 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An automated white blood cell detection and classification system using state-of-the-art deep learning, developed for deployment on mobile devices in combination with microscopes.

**Developed by:** Liam Roth
**Institution:** ETH Zurich
**Supervised by:** [Prof. Andrew deMello](https://www.demellogroup.ethz.ch/andrew-demello)
**In collaboration with:** [Prof. Stefan Balabanov](https://www.usz.ch/team/stefan-balabanov/)

---

## 🌟 Features

- **Two-Stage Detection Pipeline**: Combines YOLOv5 for fast initial detection with ResNet50 for precise classification
- **8-Class Cell Classification**: Accurately identifies Lymphocytes, RBCs, Platelets, Eosinophils, Monocytes, Band Neutrophils, Segmented Neutrophils, and Basophils
- **Multi-Scale Detection**: Image pyramid processing for handling cells at various sizes
- **Image Stitching**: Combine multiple microscope images into panoramic views for comprehensive analysis
- **Interactive GUI**: Modern PyQt6 interface with real-time visualization
- **Model Interpretability**: Integrated GradCAM for understanding model decisions
- **Production Ready**: GPU acceleration, async processing, and professional code architecture

---

## 📊 Architecture

```
Input Image
    ↓
┌─────────────────────────────┐
│ Stage 1: YOLOv5 Detection   │
│ - Fast cell localization    │
│ - Bounding box generation   │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Multi-Scale Processing      │
│ - Image pyramids (1x, 0.5x) │
│ - Sliding windows           │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Stage 2: ResNet50 Classify  │
│ - Fine-grained labeling     │
│ - Confidence scoring        │
└─────────────────────────────┘
    ↓
┌─────────────────────────────┐
│ Post-Processing             │
│ - Non-Maximum Suppression   │
│ - Box filtering & merging   │
└─────────────────────────────┘
    ↓
Cell Counts + Visualizations
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/L-Red/WBC_Counter.git
   cd WBC_Counter
   ```

2. **Clone YOLOv5 dependency**
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

3. **Install WBC Counter dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained model weights**

   Download the pre-trained weights from [Google Drive](https://drive.google.com/file/d/1M2y5-pq6S2rZP6y00y5KQ5wn27J2RMgr/view?usp=sharing) and extract them to a `weights/` directory:

   ```bash
   mkdir weights
   # Extract downloaded weights to this directory
   ```

   You should have:
   - `weights/yolo_best.pt` - YOLOv5 detection weights
   - `weights/resnet50_classifier.pt` - ResNet50 classification weights

### Running the Application

Launch the GUI application:

```bash
python app/gui_v2.py --yolo weights/yolo_best.pt --resnet weights/resnet50_classifier.pt
```

---

## 💻 Usage

### GUI Application

1. **Open Image**: Click "Open Image" to load a microscope image
2. **Capture Images**: Load multiple images for stitching
3. **Count Cells**: Click "Count Cells" to run detection and classification
4. **View Results**: Bounding boxes with dual labels (YOLO + ResNet) and cell counts

### Programmatic Usage

```python
from app.two_stage_detector import TwoStageDetector
import torch
from torchvision import transforms
from PIL import Image

# Initialize detector
detector = TwoStageDetector(
    yolo_path='weights/yolo_best.pt',
    resnet_path='weights/resnet50_classifier.pt',
    model_name='yolo'
)

# Load and preprocess image
image = Image.open('microscope_image.jpg')
transform = transforms.ToTensor()
tensor_image = transform(image)

# Run detection
results = detector.detect(tensor_image, verbose=True)

# Access results
for result in results:
    boxes = result['boxes']          # Bounding box coordinates
    labels = result['labels']        # ResNet50 classifications
    scores = result['scores']        # Confidence scores
    yolo_labels = result['yolo_labels']  # YOLO classifications
```

---

## 📁 Project Structure

```
WBC_Counter/
├── app/                          # Main application code
│   ├── gui_v2.py                 # PyQt6 GUI application
│   ├── counting_gui.py           # Legacy Tkinter interface
│   ├── two_stage_detector.py     # Two-stage detection model
│   ├── image_splitting.py        # Image processing utilities
│   ├── workers.py                # Async worker threads
│   └── evaluate_detector.ipynb   # Model evaluation notebook
│
├── torch_rcnn_try/               # Training scripts and experiments
│   ├── data_train.py             # Training data utilities
│   ├── some_funcs.py             # Helper functions
│   ├── GradCAM.py                # Model interpretability
│   ├── torch_rcnn_balanced.py    # Faster R-CNN training
│   ├── torch_resnet50_balanced.py # ResNet50 training
│   └── yolov5m.pt                # YOLOv5 base weights
│
├── image_stitching/              # Image stitching module
│   └── stitching.py              # Panorama stitching
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
├── LICENSE                       # MIT License
└── README.md                     # This file
```

---

## 🔬 Cell Types Detected

| Index | Abbreviation | Full Name              | Description                           |
|-------|--------------|------------------------|---------------------------------------|
| 0     | LY           | Lymphocytes            | Key immune cells for adaptive immunity|
| 1     | RBC          | Red Blood Cells        | Oxygen-carrying blood cells           |
| 2     | PLT          | Platelets              | Cell fragments for blood clotting     |
| 3     | EO           | Eosinophils            | Combat parasites and allergic reactions|
| 4     | MO           | Monocytes              | Largest WBCs, develop into macrophages|
| 5     | BNE          | Band Neutrophils       | Immature neutrophils                  |
| 6     | SEN          | Segmented Neutrophils  | Most abundant WBCs, first responders  |
| 7     | BA           | Basophils              | Release histamine in allergic reactions|

---

## 🎓 Training Your Own Models

### Dataset Format

Annotations should be in CSV format with columns:
- `image_path`: Path to image file
- `xmin, ymin, xmax, ymax`: Bounding box coordinates
- `class`: Cell type index (0-7)

### Training YOLOv5

```bash
cd yolov5
python train.py --img 640 --batch 16 --epochs 100 \
    --data ../data.yaml --weights yolov5m.pt
```

### Training ResNet50 Classifier

```bash
cd torch_rcnn_try
python torch_resnet50_balanced.py \
    --data_path ../datasets/centered_cells/ \
    --epochs 80 --batch_size 32
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed training instructions.

---

## 🛠️ Advanced Features

### Image Stitching

Stitch multiple microscope images into panoramas:

```python
from image_stitching.stitching import stitch_images

image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
panorama = stitch_images(image_paths)
```

### GradCAM Visualization

Visualize what regions influence the model's decisions:

```python
from torch_rcnn_try.GradCAM import generate_gradcam

gradcam_output = generate_gradcam(
    model=detector.resnet,
    image=tensor_image,
    target_class=0  # Lymphocyte
)
```

### Test-Time Augmentation

Improve prediction robustness:

```python
import ttach as tta

transforms = tta.Compose([
    tta.HorizontalFlip(),
    tta.Rotate90(angles=[0, 90, 180, 270]),
])

tta_model = tta.ClassificationTTAWrapper(detector.resnet, transforms)
```

---

## 📈 Performance

- **Detection Speed**: ~2-3 seconds per 1920x1080 image (GPU)
- **Classification Accuracy**: 94%+ on test set (varies by cell type)
- **mAP@0.5**: 0.87 for YOLOv5 detection stage
- **Supported Image Sizes**: Arbitrary (processed via sliding windows)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{roth2026wbc,
  author = {Roth, Liam},
  title = {White Blood Cell Counter: Automated WBC Detection and Classification},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/L-Red/WBC_Counter}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **YOLOv5** by [Ultralytics](https://github.com/ultralytics/yolov5)
- **PyTorch** and **torchvision** teams
- **Prof. Andrew deMello** and **Prof. Stefan Balabanov** for supervision
- ETH Zurich for institutional support

---

## 📧 Contact

**Liam Roth**
GitHub: [@L-Red](https://github.com/L-Red)

---

## 🔮 Future Improvements

- [ ] Mobile app deployment (iOS/Android)
- [ ] Real-time video stream processing
- [ ] Cloud-based batch processing
- [ ] Extended cell type classification
- [ ] Automated report generation
- [ ] Integration with laboratory information systems (LIS)

---

## ⚠️ Disclaimer

This software is intended for research and educational purposes only. It is not certified for clinical diagnostic use. Always consult qualified medical professionals for medical diagnoses.

---

**Made with ❤️ at ETH Zurich**
