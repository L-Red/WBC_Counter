from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
from torch.utils.data import DataLoader
from data_train import collate_fn, train, ResnetObjectDetectionDataset, ObjectDetectionDataset
import torch
import argparse


#use argparser for model_path and epochs
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='torch_rcnn_model.pt')
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()
model_path = args.model_path
epochs = args.epochs



# data
img_width_ratio = 1.0
img_height_ratio = 1.0
annotation_path = "../Detecto/labels_balanced.csv"
image_dir = '../datasets/centered_cells/whole_set/'
idx2name = {
    0: 'LY',
    1: 'RBC',
    2: 'PLT',
    3: 'EO',
    4: 'MO',
    5: 'BNE',
    6: 'SEN',
    7: 'BA',
    8: 'PAD',
}

name2idx = {v:k for k, v in idx2name.items()}

model = torchvision.models.resnet50(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(idx2name.keys()))

od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (1, 1), name2idx, pad=True, resize_to=(224, 224))

od_dataloader = DataLoader(od_dataset, collate_fn=collate_fn, batch_size=2)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

#split data into train and val
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(od_dataset, [0.8, 0.2], generator=generator1)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

# train model
train(model, optimizer, loss_fn, train_dataloader, val_dataloader, model_path=model_path, epochs=epochs, resnet=True, orig=True)