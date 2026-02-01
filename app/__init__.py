"""
WBC Counter - White Blood Cell Detection and Classification
============================================================

A deep learning-based system for automated detection and classification
of white blood cells from microscope images.

Main Components:
    - TwoStageDetector: Combined YOLOv5 + ResNet50 detection pipeline
    - image_splitting: Image preprocessing and multi-scale processing
    - gui_v2: PyQt6 graphical user interface
    - workers: Async worker threads for GUI operations

Example Usage:
    >>> from app.two_stage_detector import TwoStageDetector
    >>> detector = TwoStageDetector(
    ...     yolo_path='weights/yolo.pt',
    ...     resnet_path='weights/resnet.pt'
    ... )
    >>> results = detector.detect(image_tensor)

Author: Liam Roth
Institution: ETH Zurich
"""

__version__ = '1.0.0'
__author__ = 'Liam Roth'

from app.two_stage_detector import TwoStageDetector
from app import image_splitting

__all__ = ['TwoStageDetector', 'image_splitting']
