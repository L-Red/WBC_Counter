# Changelog

All notable changes to the WBC Counter project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-01

### Added
- Initial public release
- Two-stage detection pipeline (YOLOv5 + ResNet50)
- PyQt6 GUI application for interactive cell counting
- Support for 8 cell types (LY, RBC, PLT, EO, MO, BNE, SEN, BA)
- Multi-scale image processing with sliding windows
- Image stitching for panoramic microscope views
- GradCAM visualization for model interpretability
- Batch processing capabilities
- Comprehensive documentation and examples
- MIT License
- Professional README with badges and architecture diagrams
- Contributing guidelines
- Example scripts for single and batch processing
- Setup.py for pip installation
- Proper .gitignore for Python projects
- Requirements.txt with clean dependencies

### Changed
- Refactored hardcoded paths to use dynamic path resolution
- Improved YOLOv5 repository discovery
- Enhanced error handling and user feedback
- Updated documentation with clear setup instructions

### Fixed
- Fixed hardcoded model paths in two_stage_detector.py
- Corrected requirements.txt with proper pip dependencies
- Fixed missing PyQt6 and other critical dependencies

## [Unreleased]

### Planned
- Mobile app deployment (iOS/Android)
- Real-time video stream processing
- Cloud-based batch processing API
- Extended cell type classification
- Automated report generation
- Integration with laboratory information systems (LIS)
- Docker containerization
- Comprehensive test suite
- CI/CD pipeline
- Performance benchmarking tools

---

[1.0.0]: https://github.com/L-Red/WBC_Counter/releases/tag/v1.0.0
