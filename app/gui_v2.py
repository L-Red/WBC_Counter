from pathlib import Path

import numpy as np
import torch
import torchvision
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QGraphicsView, \
    QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QApplication, QProgressDialog, QGraphicsTextItem, QGridLayout, \
    QDockWidget, QSizePolicy, QDialog, QMessageBox, QErrorMessage
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QStandardPaths, QDir, pyqtSlot, QPointF, QTimer

import cv2
import sys

from skimage import exposure
from torch.nn.functional import softmax

import image_splitting  # the module containing the image processing function
from app.workers import ClassificationWorker, StitchWorker

IMAGE_WIDTH = 1200
IMAGE_HEIGHT = 800

idx2name = image_splitting.get_idx2name()


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.original_image_size = None
        self.image_path = ""

        self.setWindowTitle("Cell Counter")
        self.setGeometry(300, 300, 800, 600)

        self.individual_counts = {}  # will contain a QLabel for each class of cell

        # Set up the QGraphicsView and QGraphicsScene
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        self.capture_button = QPushButton('Capture', self)
        self.capture_button.clicked.connect(self.capture_images)


        self.button = QPushButton('Open Image', self)
        self.button.clicked.connect(self.open_image)

        self.count_button = QPushButton('Count Cells', self)
        self.count_button.clicked.connect(self.count_cells)

        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.hide()

        # create a dock widget
        self.dock = QDockWidget("Counts", self)

        # Create a wrapper QWidget
        self.dock_wrapper = QWidget()

        # create a new widget for the dock widget content
        self.dock_widget_content = QWidget(self.dock_wrapper)
        # create a new layout for the dock widget content
        self.dock_layout = QGridLayout(self.dock_widget_content)
        self.dock_layout.setHorizontalSpacing(10)  # adjust as needed
        self.dock_layout.setVerticalSpacing(2)  # adjust as needed
        self.dock_layout.setContentsMargins(5, 5, 5, 5)  # adjust these numbers as needed
        self.dock_widget_content.setLayout(self.dock_layout)

        self.total_count = QLabel("Total count: 0", self.dock_wrapper)
        self.total_count.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.dock_layout.addWidget(self.total_count, 0, 0)

        self.individual_counts = dict()
        for i, class_name in enumerate(idx2name.values(), start=1):
            self.individual_counts[class_name] = QLabel(f'{class_name}: 0', self.dock_wrapper)
            # set size policy
            self.individual_counts[class_name].setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.dock_layout.addWidget(self.individual_counts[class_name], i, 0)

        # Add dock_widget_content to wrapper layout
        self.dock_wrapper_layout = QVBoxLayout(self.dock_wrapper)
        self.dock_wrapper_layout.addWidget(self.dock_widget_content)
        self.dock_wrapper.setLayout(self.dock_wrapper_layout)

        # Set wrapper as dock widget
        self.dock.setWidget(self.dock_wrapper)

        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # makes it spin endlessly
        self.progress_bar.hide()

        # Counting label
        self.counting_label = QLabel(self)
        self.counting_label.hide()

        # Timer for '...' animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_counting_text)
        self.dots = 0

        # Set up the layout for central widget
        self.central_widget = QWidget(self)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.addWidget(self.view)  # add the QGraphicsView to the layout
        self.layout.addWidget(self.capture_button)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.count_button)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.counting_label)

        self.setCentralWidget(self.central_widget)

    def _pixmap_to_np_array(self, pixmap):
        qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_BGR888)
        byte_string = qimg.bits().asstring(qimg.sizeInBytes())
        image = np.frombuffer(byte_string, dtype=np.uint8).reshape(qimg.height(), qimg.width(), 3)
        return image

    def start_counting(self):
        self.count_button.hide()
        self.capture_button.hide()
        self.button.hide()
        self.progress_bar.show()
        self.counting_label.show()
        self.timer.start(500)  # update text every 500ms


    def animate_counting_text(self):
        self.dots = (self.dots + 1) % 4
        self.count_button.setText(f'Counting{"." * self.dots}')


    def finish_counting(self):
        # Call this function when counting is done
        self.progress_bar.hide()
        self.counting_label.hide()
        self.count_button.show()
        self.capture_button.show()
        self.button.show()
        self.timer.stop()

    def open_image(self):
        home_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        fname = QFileDialog.getOpenFileName(self, 'Open image', home_dir)
        self.image_path = fname[0]

        self.image_cv = cv2.imread(self.image_path)
        p2, p98 = np.percentile(self.image_cv, (0.5, 99.5))
        self.image_cv = exposure.rescale_intensity(self.image_cv, in_range=(p2, p98))
        
        # load the image with opencv to get its size
        image = cv2.imread(self.image_path)
        self.original_image_size = image.shape[1], image.shape[0]  # (width, height)

        p2, p98 = np.percentile(image, (0.5, 99.5))
        im_bgr = exposure.rescale_intensity(image, in_range=(p2, p98))
        qim = QImage(im_bgr.data, im_bgr.shape[1], im_bgr.shape[0], im_bgr.strides[0],
                     QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qim)

        self.display_image(pixmap)

    def count_cells(self):
        try:
            image = cv2.cvtColor(self.image_cv, cv2.COLOR_BGR2RGB)
            self.worker = ClassificationWorker(image, self.original_image_size)
            self.worker.signal_pixmap.connect(self.display_image)
            self.worker.signal_progress.connect(self.progress.setValue)
            self.worker.signal_boxes_labels.connect(self.draw_boxes_labels)
            self.worker.signal_counts.connect(self.update_counts)
            self.worker.finished.connect(self.counting_finished)
            self.worker.signal_finish.connect(self.finish_counting)
            self.progress.show()
            self.worker.start()
            self.start_counting()
        except AttributeError:
            # If error occurs during stitching
            self.finish_counting()
            error_dialog = QErrorMessage(self)
            error_dialog.setWindowTitle("Error!")
            error_dialog.showMessage("An image must be opened before counting cells.")
            error_dialog.exec()

    def stitch_images(self, image_paths):
        self.progress.show()
        self.worker = StitchWorker(image_paths)
        self.worker.signal_pixmap.connect(self.display_image)
        self.worker.signal_progress.connect(self.progress.setValue)
        self.worker.signal_error.connect(self.handle_error)
        self.worker.start()

    def capture_images(self):
        home_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.PicturesLocation)
        dir_name = QFileDialog.getExistingDirectory(self, 'Select Directory', home_dir)

        if not dir_name:
            return

        # Get all image files in the directory
        fnames = [str(path) for path in     Path(dir_name).glob('*') if
                  path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]

        if not fnames:
            return
        # Show the preview window
        preview_dialog = QDialog(self)
        preview_layout = QGridLayout(preview_dialog)

        for i, fname in enumerate(fnames):
            pixmap = QPixmap(fname)
            smaller_pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            label = QLabel()
            label.setPixmap(smaller_pixmap)
            preview_layout.addWidget(label, i // 5, i % 5)  # adjust grid layout as per your needs

        ok_button = QPushButton('OK', preview_dialog)
        ok_button.clicked.connect(preview_dialog.accept)
        preview_layout.addWidget(ok_button, len(fnames) // 5 + 1, 0)

        if preview_dialog.exec() == QDialog.DialogCode.Accepted:
            self.stitch_images(fnames)

    @pyqtSlot(QPixmap)
    def display_image(self, pixmap):
        self.image_cv = self._pixmap_to_np_array(pixmap)
        self.original_image_size = self.image_cv.shape[1], self.image_cv.shape[0]
        smaller_pixmap = pixmap.scaled(IMAGE_WIDTH, IMAGE_HEIGHT, Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.clear()
        self.scene.addPixmap(smaller_pixmap)
        self.actual_image_width = smaller_pixmap.width()
        self.actual_image_height = smaller_pixmap.height()

    @pyqtSlot(list)
    def draw_boxes_labels(self, boxes_labels):
        boxes_labels = boxes_labels[0]
        # get the scaling factors
        orig_width, orig_height = self.worker.original_image_size
        disp_width, disp_height = self.actual_image_width, self.actual_image_height
        scale_width, scale_height = disp_width / orig_width, disp_height / orig_height

        # Here you draw boxes and labels on your image
        for i, box in enumerate(boxes_labels['boxes']):
            xmin, ymin, xmax, ymax = box
            xmin_scaled, ymin_scaled = xmin * scale_width, ymin * scale_height
            xmax_scaled, ymax_scaled = xmax * scale_width, ymax * scale_height

            # For yolo_labels and yolo_scores, assuming they are structured in a similar way as "boxes" and "labels"
            yolo_label = boxes_labels["yolo_labels"][i]
            yolo_score = boxes_labels["yolo_scores"][i]

            label = boxes_labels["labels"][i]
            score = boxes_labels["scores"][i]

            # Assuming box coordinates are (x1, y1, x2, y2)
            self.scene.addRect(xmin_scaled, ymin_scaled, xmax_scaled - xmin_scaled, ymax_scaled - ymin_scaled)

            # You can also add label text next to the box with different colors
            text1 = QGraphicsTextItem(f'{label}')
            text1.setDefaultTextColor(QColor("blue"))
            text1.setPos(xmin_scaled, ymin_scaled)

            text2 = QGraphicsTextItem(f'YOLO Label: {yolo_label}')
            text2.setDefaultTextColor(QColor("red"))
            text2.setPos(xmin_scaled, ymax_scaled)

            self.scene.addItem(text1)
            self.scene.addItem(text2)

    @pyqtSlot(int, dict)
    def update_counts(self, total_wbc, count_dict):
        # Clear the current counts
        for i in reversed(range(self.dock_layout.count())):
            self.dock_layout.itemAt(i).widget().setParent(None)

        # Create and add new counts
        self.total_wbc_count = QLabel(f'Total WBC Count: {total_wbc}')
        self.dock_layout.addWidget(self.total_wbc_count)
        for class_name, count in count_dict.items():
            if class_name == "total": continue  # skip the total count
            self.individual_counts[class_name] = QLabel(f'{class_name}: {count}')
            self.dock_layout.addWidget(self.individual_counts[class_name])

    def counting_finished(self):
        self.progress.hide()

    def handle_error(self, error_message):
        QMessageBox.critical(self, 'Error', f'Sorry! An error has occurred: {error_message}')




if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
