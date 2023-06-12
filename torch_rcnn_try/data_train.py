import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.utils
import torch.utils.checkpoint
from bbaug import policies
from torchvision import transforms
import pandas as pd

from torchvision.ops import clip_boxes_to_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# make image augmentor class
class ImageAugmentor():
    def __init__(self):
        self.aug_policy = policies.policies_v2()
        self.policy_container = policies.PolicyContainer(self.aug_policy)
    def __call__(self, image_batch, targets_batch, resnet=False):

        # select a random policy from the policy set
        random_policy = self.policy_container.select_random_policy()
        aug_images = []
        aug_targets = []
        for idx, image in enumerate(image_batch):

            bounding_boxes = targets_batch[idx]['boxes']
            labels = targets_batch[idx]['labels']

            image = image.cpu().permute(1, 2, 0).numpy()
            img_shape = image.shape[:2]
            # image from float32 to uint8
            image = (image * 255).astype(np.uint8)
            bounding_boxes = bounding_boxes.numpy()
            labels = labels.numpy()

            img_aug, bbs_aug = self.policy_container.apply_augmentation(random_policy, image, bounding_boxes, labels)
            # image from uint8 to float32
            img_aug = img_aug.astype(np.float32) / 255

            # bbs_aug: numpy array of augmneted bounding boxes in format: [[label, x_min, y_min, x_man, y_max],...]
            # split bbs_aug into labels and bounding boxes
            if len(bbs_aug.shape) != 2:
                if resnet:
                    #invent some bbs_aug
                    bbs_aug = np.array([[0, 0, 0, 0, 0]])
                else:
                    continue
            labels_aug = bbs_aug[:, 0]
            bbs_aug = bbs_aug[:, 1:]
            # convert bounding boxes from numpy array to tensor
            bbs_aug = torch.from_numpy(bbs_aug)
            # clip bounding boxes to image dimensions
            bbs_aug = clip_boxes_to_image(bbs_aug, img_shape)
            # image to tensor
            img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).to(device=device)

            # remove bboxes where width or height is <= 0
            # get width and height of bounding boxes
            width = bbs_aug[:, 2] - bbs_aug[:, 0]
            height = bbs_aug[:, 3] - bbs_aug[:, 1]
            wh_idx = torch.logical_and((width > 0), (height > 0))
            wh_idx = torch.where(wh_idx)[0]
            # get bounding boxes where width and height are > 0
            bbs_aug = bbs_aug[wh_idx]
            # get labels where width and height are > 0
            labels_aug = labels_aug[wh_idx]

            if type(labels_aug) == np.int32:
                labels_aug = np.array([labels_aug], dtype=np.int64)
            # convert bounding boxes to tensor
            # labels to tensor
            labels_aug = torch.from_numpy(labels_aug)
            labels_aug = labels_aug.to(dtype=torch.int64)
            # append augmented image and targets
            aug_images.append(img_aug)
            aug_targets.append({'boxes': bbs_aug, 'labels': labels_aug})
        # convert augmented images to tuple
        aug_images = tuple(aug_images)
        # convert augmented targets to tuple
        aug_targets = tuple(aug_targets)
        return aug_images, aug_targets

def parse_annotation(annotation_path, image_dir, image_size_ratio, rbc=True):
    """Parse the annotations file."""

    df = pd.read_csv(annotation_path)
    gt_boxes_all = []
    gt_classes_all = []
    img_paths = []
    for file in df['filename'].unique():
        img_path = os.path.join(image_dir, file)
        # give df with filename == file
        df_file = df[df['filename'] == file]
        gt_classes = []
        gt_boxes = []
        for index, row in df_file.iterrows():
            gt_class = row['class']
            if not rbc:
                if gt_class == 'RBC':
                    continue
            w = row['width']
            h = row['height']
            gt_box = row[['xmin', 'ymin', 'xmax', 'ymax']].values
            # resize the bounding boxes to the new image size where image_size is (width, height)
            # gt_box = gt_box / np.array([w, h, w, h])
            gt_box = gt_box * np.array([image_size_ratio[1], image_size_ratio[0], image_size_ratio[1], image_size_ratio[0]])
            gt_box = gt_box.astype(np.int32)
            gt_box = gt_box
            gt_classes.append(gt_class)
            gt_boxes.append(gt_box)
        gt_boxes = np.array(gt_boxes)
        img_paths.append(img_path)
        gt_boxes_all.append(torch.tensor(gt_boxes, dtype=torch.float32))
        gt_classes_all.append(gt_classes)


    return gt_boxes_all, gt_classes_all, img_paths


class ObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.

Parameters
    ------------
    annotation_path: str
        path to the annotation file
    img_dir: str
        path to the directory containing the images
    img_size_ratio: tuple
        tuple of (height, width) to resize the images to

    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''

    def __init__(self, annotation_path, img_dir, img_size_ratio, name2idx, pad=True, resize_to=None, rbc=True):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size_ratio = img_size_ratio
        self.name2idx = name2idx
        self.resize_to = resize_to
        self.rbc = rbc
        self.img_data_all, self.targets = self.get_data(pad)

    def __len__(self):
        return len(self.img_data_all)

    def __getitem__(self, idx):
        return self.img_data_all[idx], self.targets[idx]

    def get_data(self, pad):
        img_data_all = []
        gt_idxs_all = []
        targets = []

        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size_ratio, rbc=self.rbc)


        for i, img_path in enumerate(img_paths):

            # skip if the image path is not valid
            if (not img_path) or (not os.path.exists(img_path)):
                continue

            # read and resize image
            img = io.imread(img_path)
            if self.resize_to is None:
                new_size = (int(img.shape[0] * self.img_size_ratio[0]), int(img.shape[1] * self.img_size_ratio[1]))
            else:
                new_size = self.resize_to
            img = resize(img, new_size, anti_aliasing=True)

            # augment image


            # convert image to torch tensor and reshape it so channels come first
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)

            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])



            img_data_all.append(img_tensor.to(dtype=torch.float32))
            gt_idxs_all.append(gt_idx)

        if pad:
            # pad bounding boxes and classes so they are of the same size
            gt_bboxes_pad = pad_sequence(gt_boxes_all, batch_first=True, padding_value=self.name2idx['PAD'])
            gt_classes_pad = pad_sequence(gt_idxs_all, batch_first=True, padding_value=self.name2idx['PAD'])
        else:
            gt_bboxes_pad = gt_boxes_all
            gt_classes_pad = gt_idxs_all

        for i in range(len(gt_classes_pad)):
            # create target dictionary
            target = {}
            target['boxes'] = gt_bboxes_pad[i].to(dtype=torch.int64)
            target['labels'] = gt_classes_pad[i].to(dtype=torch.int64)
            targets.append(target)

        # stack all images
        # img_data_stacked = torch.stack(img_data_all, dim=0)

        return img_data_all, targets

from skimage import io
import os
from skimage.transform import resize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch


class ResnetObjectDetectionDataset(Dataset):
    '''
    A Pytorch Dataset class to load the images and their corresponding annotations.

    Returns
    ------------
    images: torch.Tensor of size (B, C, H, W)
    gt bboxes: torch.Tensor of size (B, max_objects, 4)
    gt classes: torch.Tensor of size (B, max_objects)
    '''

    def __init__(self, annotation_path, img_dir, img_size_ratio, name2idx, pad=True):
        self.annotation_path = annotation_path
        self.img_dir = img_dir
        self.img_size_ratio = img_size_ratio
        self.name2idx = name2idx

        self.img_data_all, self.targets = self.get_data(pad)

    def __len__(self):
        return self.img_data_all.size(dim=0)

    def __getitem__(self, idx):
        return self.img_data_all[idx], self.targets[idx]

    def get_data(self, pad):
        img_data_all = []
        gt_idxs_all = []
        targets = []

        gt_boxes_all, gt_classes_all, img_paths = parse_annotation(self.annotation_path, self.img_dir, self.img_size_ratio)

        for i, img_path in enumerate(img_paths):

            # skip if the image path is not valid
            if (not img_path) or (not os.path.exists(img_path)):
                continue

            # read and resize image
            img = io.imread(img_path)
            # new_size = (256, 256)
            # img = resize(img, new_size, anti_aliasing=True)

            # encode class names as integers
            gt_classes = gt_classes_all[i]
            gt_idx = torch.Tensor([self.name2idx[name] for name in gt_classes])

            # for each bbox, cut out the image and resize it to 256x256
            for j, bbox in enumerate(gt_boxes_all[i]):
                bbox = bbox.to(dtype=torch.int32)
                x1, y1, x2, y2 = bbox
                img_cropped = img[y1:y2, x1:x2]
                img_cropped = resize(img_cropped, (256, 256), anti_aliasing=True)
                img_tensor = torch.from_numpy(img_cropped).permute(2, 0, 1)
                img_data_all.append(img_tensor)
                gt_idxs_all.append(gt_idx[j].item())



        # stack all images
        img_data_stacked = torch.stack(img_data_all, dim=0)
        targets = torch.tensor(gt_idxs_all)

        return img_data_stacked.to(dtype=torch.float32), targets.to(dtype=torch.long)

def collate_fn(batch):
    return tuple(zip(*batch))

def orig_data_to_resnet(img_batch, targets_batch):
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

    name2idx = {v: k for k, v in idx2name.items()}
    # move data to device
    img_batch = torch.stack(img_batch).to(device)
    new_targets_batch = []
    for target in targets_batch:
        labels = target['labels']
        # give all indeces in labels where entry is not 'PAD' or 'RBC'
        idxs = [i for i, x in enumerate(labels) if x != name2idx['PAD'] and x != name2idx['RBC']]
        if len(idxs) == 0:
            new_targets_batch.append(labels[0])
        else:
            if len(idxs) > 1:
                # pick one that is not 'PLT' if possible
                for idx in idxs:
                    if labels[idx] != name2idx['PLT']:
                        new_target = labels[idx]
                        break
                    else:
                        new_target = labels[idxs[0]]
                new_targets_batch.append(new_target)

            else:
                new_targets_batch.append(labels[idxs[0]])
    targets_batch = torch.tensor(new_targets_batch)
    targets_batch = targets_batch.to(device)
    return img_batch, targets_batch

# define a training loop per batch with backpropagation
def train_one_batch(img_batch, targets_batch, model, optimizer, loss_fn):
    # move data to device
    img_batch = [img.to(device) for img in img_batch]
    targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    # outputs = model(img_batch, targets_batch)
    # checkpointing
    outputs = torch.utils.checkpoint.checkpoint(model, img_batch, targets_batch)
    # weigh classification loss by 10
    # for v in outputs.values():
    #     # run backprop
    #     v.backward()
    # outputs['loss_classifier'] *= 10
    loss = sum(sum(loss) for loss in outputs.values())

    # running_loss = loss.item() * len(targets_batch)
    # running_corrects = torch.sum(preds == targets_batch.data)

    # loss.backward()
    optimizer.step()

    return loss

def train_one_batch_resnet(img_batch, targets_batch, model, optimizer, loss_fn):
    # move data to device
    img_batch = img_batch.to(device)
    targets_batch = targets_batch.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled(True):
        outputs = model(img_batch)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, targets_batch)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

    return preds.cpu().numpy(), targets_batch.cpu().numpy()
# define a validation loop per batch
def validate_one_batch(img_batch, targets_batch, model, loss_fn, map):
    # Initialize metric

    # move data to device
    img_batch = [img.to(device) for img in img_batch]
    targets_batch = [{k: v.to(device) for k, v in t.items()} for t in targets_batch]

    # forward
    # outputs = model(img_batch)
    outputs = torch.utils.checkpoint.checkpoint(model, img_batch)
    map.update(outputs, targets_batch)
    # Compute the results
    map_result = map.compute()





    # calculate classification accuracy with 'labels' key
    acc = 0
    output_labels = []
    true_labels = []
    for idx, output in enumerate(outputs):
        labels_true = targets_batch[idx]['labels']
        labels_pred = output['labels']
        # extend shorter tensor with pad(8) so that they are of the same size
        if len(labels_true) < len(labels_pred):
            labels_true = torch.cat((labels_true, torch.Tensor([8] * (len(labels_pred) - len(labels_true))).to(device)))
        elif len(labels_true) > len(labels_pred):
            labels_pred = torch.cat((labels_pred, torch.Tensor([8] * (len(labels_true) - len(labels_pred))).to(device)))
        output_labels.append(labels_pred)
        true_labels.append(labels_true)
        # calculate accuracy
        acc += (labels_true == labels_pred).sum().item() / len(labels_true)
    # calculate loss
    output_labels = torch.cat(output_labels)
    true_labels = torch.cat(true_labels)
    output_labels = output_labels.to(torch.float32)
    true_labels = true_labels.to(torch.float32)
    running_loss = loss_fn(output_labels, true_labels)

    return running_loss, acc / len(outputs), map_result

def validate_one_batch_resnet(img_batch, targets_batch, model, optimizer, loss_fn):
    # move data to device
    img_batch = img_batch.to(device)
    targets_batch = targets_batch.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward
    # track history if only in train
    with torch.set_grad_enabled(False):
        outputs = model(img_batch)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, targets_batch)

    return preds.cpu().numpy(), targets_batch.cpu().numpy()

# define a training loop per epoch
def train_one_epoch(model, optimizer, loss_fn, dataloader, resnet, orig, scheduler=None):
    # set model to training mode
    model.train()

    # augmentor
    aug = ImageAugmentor()
    if resnet:
        running_loss = 0.0
        running_corrects = 0

        # transforms
        transform_arr = transforms.Compose([
            transforms.RandomResizedCrop(240),
            transforms.Resize(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    all_targets = []
    all_preds = []
    total_loss = 0
    # iterate over batches
    for img_batch, targets_batch in dataloader:
        # augment images
        if resnet:
            if orig:
                img_batch, targets_batch = orig_data_to_resnet(img_batch, targets_batch)

            img_batch = [transform_arr(transforms.ToPILImage()(img)) for img in img_batch]
            img_batch = torch.stack([img / img.max() for img in img_batch])
            img_batch = img_batch.to(device)

            preds, targets = train_one_batch_resnet(img_batch, targets_batch, model, optimizer, loss_fn)
            all_preds.extend(preds)
            all_targets.extend(targets)
            total_loss += running_loss
        else:
            img_batch, targets_batch = aug(img_batch, targets_batch)
            loss = train_one_batch(img_batch, targets_batch, model, optimizer, loss_fn)
            total_loss += loss.item()
    if resnet:
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets, all_preds)
        conf_mat = confusion_matrix(all_targets, all_preds)

        print('Validation Results:')
        print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} Accuracy: {accuracy:.4f}')

        if scheduler is not None:
            scheduler.step()

    return total_loss

# define a validation loop per epoch
def validate_one_epoch(model, loss_fn, dataloader, optimizer, resnet, orig, best_acc):
    # set model to evaluation mode
    model.eval()

    # augmentor
    if resnet:
        running_loss = 0.0
        running_corrects = 0
        transform_arr = transforms.Compose([
            transforms.CenterCrop(240),
            transforms.Resize(299),
            transforms.ToTensor(),
        ])
    map_module = MeanAveragePrecision(iou_type="bbox")
    total_loss = 0
    total_acc = 0
    total_map = 0
    total_mar = 0
    all_preds = []
    all_targets = []
    # iterate over batches
    for img_batch, targets_batch in dataloader:
        if resnet:
            if orig:
                img_batch, targets_batch = orig_data_to_resnet(img_batch, targets_batch)

            img_batch = [transform_arr(transforms.ToPILImage()(img)) for img in img_batch]
            img_batch = torch.stack([img / img.max() for img in img_batch])
            img_batch = img_batch.to(device)

            preds, targets = validate_one_batch_resnet(img_batch, targets_batch, model, optimizer, loss_fn)
            all_preds.extend(preds)
            all_targets.extend(targets)
        else:
            loss, acc, map_result = validate_one_batch(img_batch, targets_batch, model, loss_fn, map_module)
            total_loss += loss.item()
            total_acc += acc
            map, mar = map_result['map'], map_result['mar_1']
            total_map += map
            total_mar += mar


    if resnet:

        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        accuracy = accuracy_score(all_targets, all_preds)
        conf_mat = confusion_matrix(all_targets, all_preds)

        print('Validation Results:')
        print(f'Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f} Accuracy: {accuracy:.4f}')

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), './resnet50_best.pt')
        return precision, recall, f1, accuracy, conf_mat, best_acc
    else:
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = total_acc / len(dataloader.dataset)
        epoch_map = total_map / len(dataloader.dataset)
        epoch_mar = total_mar / len(dataloader.dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} mAP: {epoch_map:.4f} mAR: {epoch_mar:.4f}')

        if epoch_map > best_acc:
            best_acc = epoch_map
            torch.save(model.state_dict(), './best_rcnn.pt')

        return total_loss, total_acc / len(dataloader), best_acc


# define a training loop
def train(model, optimizer, loss_fn, train_dataloader, val_dataloader, model_path, epochs, resnet=False, orig=False):
    # move model to device
    model.to(device)
    # if device.type == 'cuda':
    #     model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # initialize lists for plotting progress
    train_losses = []
    val_losses = []
    val_accs = []

    epoch_precision = []
    epoch_recall = []
    epoch_f1 = []
    epoch_accuracy = []
    epoch_conf_mat = []

    print('Start training...')

    best_acc = 0.0

    if resnet:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    else:
        scheduler = None

    # iterate over epochs
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        # train model for one epoch
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_dataloader, resnet=resnet, orig=orig, scheduler=scheduler)


        # validate model
        if resnet:
            precision, recall, f1, accuracy, conf_mat, best_acc = validate_one_epoch(model, loss_fn, val_dataloader, optimizer, resnet=resnet, orig=orig, best_acc=best_acc)
            epoch_precision.append(precision)
            epoch_recall.append(recall)
            epoch_f1.append(f1)
            epoch_accuracy.append(accuracy)
            epoch_conf_mat.append(conf_mat)
        else:
            val_loss, val_acc, best_acc = validate_one_epoch(model, loss_fn, val_dataloader, optimizer, resnet=resnet, orig=orig, best_acc=best_acc)

            # save losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)


    if resnet:
        # make and save plots with metrics over epochs using grid for each metric
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].plot(epoch_precision)
        axs[0, 0].set_title('Precision')
        axs[0, 1].plot(epoch_recall, 'tab:orange')
        axs[0, 1].set_title('Recall')
        axs[1, 0].plot(epoch_f1, 'tab:green')
        axs[1, 0].set_title('F1')
        axs[1, 1].plot(epoch_accuracy, 'tab:red')
        axs[1, 1].set_title('Accuracy')
        # save plot
        plt.savefig('metrics.png')
        plt.close()
    # plot losses
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.plot(val_accs, label='val acc')
    plt.legend()
    # save plot
    plt.savefig('loss.png')
    plt.close()

    #save model
    torch.save(model.state_dict(), model_path)