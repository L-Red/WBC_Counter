from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision
from torch.utils.data import DataLoader
from data_train import ObjectDetectionDataset, train, ResnetObjectDetectionDataset
from data_train import collate_fn as collate_rcnn
import torch
import argparse



#use argparser for model_path and epochs
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='torch_rcnn_model.pt')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--pretrained', type=int, default=0)
parser.add_argument('--rbc', type=int, default=1)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str)
parser.add_argument('--data_parallel', type=int, default=0)
parser.add_argument('--optimizer', type=str, default='adam')
args = parser.parse_args()
model_path = args.model_path
epochs = args.epochs
pretrained = bool(args.pretrained)
rbc = bool(args.rbc)
model_name = args.model
data = args.data
data_parallel = bool(args.data_parallel)
opt_name = args.optimizer



# data
img_width_ratio = 1.0
img_height_ratio = 1.0
annotation_path = "../Detecto/labels_balanced.csv"
image_dir = '../datasets/centered_cells/whole_set/'
pad = False
idx2name = {
    0: 'LY',
    1: 'RBC',
    2: 'PLT',
    3: 'EO',
    4: 'MO',
    5: 'BNE',
    6: 'SEN',
    7: 'BA'
            }

if pad:
    idx2name[-1] = 'PAD'

name2idx = {v:k for k, v in idx2name.items()}

is_orig = False

if model_name == 'resnet':
    is_resnet = True
    model = torchvision.models.resnet50(weights='DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(idx2name.keys()))
    if data == 'orig':
        is_orig = True
        od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (1, 1), name2idx, pad=True, resize_to=(224, 224))
        collate_fn = collate_rcnn
    else:
        od_dataset = ResnetObjectDetectionDataset(annotation_path, image_dir, (img_height_ratio, img_width_ratio), name2idx, pad=False)
        collate_fn = None
elif model_name == 'rcnn':
    is_resnet = False
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=len(idx2name.keys()))

    od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height_ratio, img_width_ratio), name2idx, pad=False)
    collate_fn = collate_rcnn

od_dataloader = DataLoader(od_dataset, collate_fn=collate_fn, batch_size=2)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

if data_parallel:
    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids)


# define optimizer and loss function
if opt_name == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif opt_name == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

#split data into train and val
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(od_dataset, [0.8, 0.2], generator=generator1)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=2)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

# train model
train(model, optimizer, loss_fn, train_dataloader, val_dataloader, model_path=model_path, epochs=epochs, resnet=is_resnet, orig=is_orig)