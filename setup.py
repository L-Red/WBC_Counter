"""Setup script for WBC Counter package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="wbc-counter",
    version="1.0.0",
    author="Liam Roth",
    author_email="",
    description="Automated White Blood Cell Counter using Deep Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/L-Red/WBC_Counter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.6.0",
        "scikit-image>=0.19.0",
        "Pillow>=9.2.0",
        "PyQt6>=6.4.0",
        "stitching>=0.1.0",
        "numpy>=1.24.0",
        "pandas>=1.4.0",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "matplotlib>=3.5.0",
        "grad-cam==1.4.6",
        "ttach==0.0.3",
        "bbaug>=0.1.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "training": [
            "tensorboard>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wbc-counter=app.gui_v2:main",
        ],
    },
    include_package_data=True,
    keywords="white blood cell detection classification medical imaging deep learning computer vision",
)
