"""workers.py: This file defines the worker threads for the PyQT6 front end appliction. It launches the different subroutines necessary for the blood cell count."""

import cv2
import torch
import torchvision
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QErrorMessage
from torch.nn.functional import softmax

from app import image_splitting
from app.two_stage_detector import TwoStageDetector
from image_stitching.stitching import MyStitcher

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

MODEL_NAME = 'yolo'
# if MODEL_NAME == 'rcnn':
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
#     #load weights
#     model.load_state_dict(torch.load('../torch_rcnn_try/runs/run7_noweights/model_run4_balance_v2_80_noweights.pt', map_location=device))
# elif MODEL_NAME == 'yolo':
#     # Load the saved file
#     model = torch.hub.load('../yolov5', 'custom', path='../yolov5/runs/euler/exp1/weights/best.pt', source='local')  # local model

idx2name = image_splitting.get_idx2name()

# resnet = torchvision.models.resnet50()
# num_ftrs = resnet.fc.in_features
# resnet.fc = torch.nn.Linear(num_ftrs, len(idx2name.keys()))
# resnet.load_state_dict(torch.load('../torch_rcnn_try/runs/run15_last_resnet/resnet50_best.pt', map_location=device))


class StitchWorker(QThread):
    signal_pixmap = pyqtSignal(QPixmap)
    signal_progress = pyqtSignal(int)
    signal_error = pyqtSignal(str)

    def __init__(self, image_paths):
        QThread.__init__(self)
        self.image_paths = image_paths
        self.stitcher = MyStitcher()

    def run(self):
        try:
            print("Stitching images...")
            # Assuming you have a function stitch_images in your image_splitting module
            stitched_image = self.stitcher(self.image_paths)

            # Convert the stitched image to QImage
            stitched_image_cv = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB)
            qim = QImage(stitched_image_cv.data, stitched_image_cv.shape[1], stitched_image_cv.shape[0], stitched_image_cv.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qim)
            self.signal_pixmap.emit(pixmap)
        except Exception as e:
            self.signal_error.emit(str(e))
            print(e)

class ClassificationWorker(QThread):
    signal_pixmap = pyqtSignal(QPixmap)
    signal_boxes_labels = pyqtSignal(list)
    signal_progress = pyqtSignal(int)
    signal_counts = pyqtSignal(int, dict)
    signal_finish = pyqtSignal()

    def __init__(self, image, original_image_size, resnet, yolo, model_name=MODEL_NAME):
        QThread.__init__(self)
        self.image = image
        self.original_image_size = original_image_size
        self.model_name = model_name
        self.two_stage_detector = TwoStageDetector(yolo_path=yolo, resnet_path=resnet)


    def run(self):
        try:
            boxes_labels, total_wbc, individual_counts = self._r()
            self.signal_boxes_labels.emit(boxes_labels)
            self.signal_counts.emit(total_wbc, individual_counts)
            self.signal_finish.emit()
        except:
            # If error occurs during stitching
            error_dialog = QErrorMessage(self)
            error_dialog.setWindowTitle("Error!")
            error_dialog.showMessage("Something went wrong during cell counting. Please try again.")
            error_dialog.exec()
            self.signal_finish.emit()

    def _r(self):
        tensor_image = torch.Tensor(self.image).permute(2, 0, 1)
        # Run the image processing function
        outputs = self.two_stage_detector.detect(tensor_image, verbose=True)

        # Convert image back to QImage to emit signal
        # im_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        im_bgr = self.image
        qim = QImage(im_bgr.data, im_bgr.shape[1], im_bgr.shape[0], im_bgr.strides[0],
                     QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qim)
        self.signal_pixmap.emit(pixmap)

        # Count the number of WBCs
        total_wbc = 0
        individual_counts = {}
        for output in outputs:
            total_wbc += len(output['labels'])
            for label in output['labels']:
                if label in individual_counts:
                    individual_counts[label] += 1
                else:
                    individual_counts[label] = 1

        # Emit boxes and labels
        boxes_labels = outputs
        return boxes_labels, total_wbc, individual_counts
