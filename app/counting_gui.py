"""
counting_gui.py: In this code, we implement running the model from the front-end PyQt6 application. 
It let's the user open an image, it blocks usage during inference and updates the final image with the resulting bounding boxes after.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from PIL import Image, ImageTk, ImageDraw, ImageFont
from threading import Thread

from torchvision import transforms

import image_splitting
import queue

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
# #load weights
# model.load_state_dict(torch.load('../torch_rcnn_try/runs/run3/model_run3_40.pt', map_location=torch.device('cpu')))

# v2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
#load weights
model.load_state_dict(torch.load('../torch_rcnn_try/runs/run4/model_run4_balance_v2_40.pt', map_location=torch.device('cpu')))


def open_image():
    global image

    to_tensor = transforms.ToTensor()
    file_path = filedialog.askopenfilename()
    image = Image.open(file_path)
    photo = image.resize((250, 250), Image.LANCZOS)  # resize for the demo
    photo = ImageTk.PhotoImage(photo)
    label.config(image=photo)
    label.image = photo  # keep a reference to the image

    image = to_tensor(image)


def count_cells():
    # create a progress bar
    progress = ttk.Progressbar(root, length=300, mode='indeterminate')
    progress.pack()

    # create a thread that will run the counting function
    thread = Thread(target=run_inference, args=(image, progress))
    thread.start()


def draw_boxes(image, results):
    # Convert image tensor back to PIL image
    image = transforms.ToPILImage()(image.squeeze())
    draw = ImageDraw.Draw(image)

    for result in results:
        boxes = result['boxes']
        labels = result['labels']
        scores = result['scores']
        for box, label, score in zip(boxes, labels, scores):
            # Draw a rectangle
            draw.rectangle(box.tolist(), outline='red', width=3)

            # Draw label and score
            draw.text((box[0], box[1] - 10), f'{label}: {score:.2f}', fill='red')

    return image


def show_result(image, results):

    print(f'Results: {results}')

    result_image = draw_boxes(image, results)
    # Resize the image
    result_image = result_image.resize((250,250), Image.LANCZOS)
    result_image = ImageTk.PhotoImage(result_image)
    label.configure(image=result_image)
    label.image = result_image
    progress.place_forget()  # hide the progress bar


def run_inference(image, progress):
    # start the progress bar
    gui_queue.put(("progress", "start"))

    # run the cell counting
    results = image_splitting.split_inference_reconstruct(model, image)

    # stop the progress bar
    gui_queue.put(("progress", "stop"))

    # pass the results to the main thread
    gui_queue.put(("result", (image, results)))


def update_gui():
    while not gui_queue.empty():
        try:
            message = gui_queue.get(0)
            if message[0] == "progress":
                if message[1] == "start":
                    progress.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # show the progress bar
                    progress.start()
                elif message[1] == "stop":
                    progress.stop()
            elif message[0] == "result":
                show_result(*message[1])
        except queue.Empty:
            pass

    # schedule the next update
    root.after(100, update_gui)  # every 100 ms



root = tk.Tk()
gui_queue = queue.Queue()

button_open = tk.Button(root, text="Open Image", command=open_image)
button_open.pack()

label = tk.Label(root)
label.pack()

button_count = tk.Button(root, text="Count Cells", command=count_cells)
button_count.pack()

progress = ttk.Progressbar(root, length=300, mode='indeterminate')
progress.pack()

update_gui()  # start updating the GUI
root.mainloop()
