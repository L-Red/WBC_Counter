# White Blood Cell Counter

This is a blood cell counter for the applied use on a mobile device in combination with a microscope developed by Liam Roth. 
This is developed as a semester project for [Prof. Andrew deMello](https://www.demellogroup.ethz.ch/andrew-demello) at ETH Zurich. This is done in collaboration with [Prof. Stefan Balabanov](https://www.usz.ch/team/stefan-balabanov/).

## Installation

The tool at the moment makes use of the YoloV5 model. Since it has a separate git repository and is quite large, one has to 
install this repository in the following manner:

```bash
git clone https://github.com/L-Red/blood_cell_counter.git
cd blood_cell_counter
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

This repository has just been created. I doubt it runs smoothly for everyone everywhere but I will soon extend this to make it more accessible for more people.

You can run the gui application after installing the `requirements.txt` by running

```bash
python app/gui_v2.py
```

If something doesn't work, feel free to let me know! :)
