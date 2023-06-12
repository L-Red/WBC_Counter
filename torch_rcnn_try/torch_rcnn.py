from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
from torch.utils.data import DataLoader
from data_train import ObjectDetectionDataset, collate_fn, train
import torch
import argparse


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

#use argparser for model_path and epochs
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='torch_rcnn_model.pt')
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()
model_path = args.model_path
epochs = args.epochs



# data
img_width_ratio = 640/360
img_height_ratio = 480/360
annotation_path = "../Detecto/labels.csv"
image_dir = '../datasets/centered_cells/whole_set/'
idx2name = {
    -1: 'PAD',
    0: 'LY',
    1: 'RBC',
    2: 'PLT',
    3: 'EO',
    4: 'MO',
    5: 'BNE',
    6: 'SEN',
    7: 'BA'
            }

name2idx = {v:k for k, v in idx2name.items()}

od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height_ratio, img_width_ratio), name2idx, pad=False)

od_dataloader = DataLoader(od_dataset, collate_fn=collate_fn, batch_size=2)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

# train model
train(model, optimizer, loss_fn, od_dataloader, od_dataloader, model_path=model_path, epochs=epochs)