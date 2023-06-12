import os

import pandas as pd
import numpy as np
import torch
import torchvision
from matplotlib import patches
from torch import nn
from torchvision import ops
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

def score_output(output, ground_truth, iou_threshold=0.5):
    precisions = []

    num_classes = max([max(d['labels'].max() for d in output), max(d['labels'].max() for d in ground_truth)]) + 1

    for c in range(num_classes):
        # Get boxes and labels for this class
        true_boxes_class = torch.cat([d['boxes'][d['labels'] == c] for d in ground_truth])
        predicted_boxes_class = torch.cat([d['boxes'][d['labels'] == c] for d in output])

        if true_boxes_class.shape[0] == 0:
            continue  # No ground truth for this class

        if predicted_boxes_class.shape[0] == 0:
            precisions.append(0)  # No prediction for this class
            continue

        # Apply NMS to predicted boxes
        keep_indices = torch.ops.torchvision.nms(predicted_boxes_class, torch.ones(predicted_boxes_class.shape[0]),
                                                 iou_threshold)
        predicted_boxes_class = predicted_boxes_class[keep_indices]

        tp = 0
        # Iterate over predicted boxes
        for pb in predicted_boxes_class:
            # Calculate IoU with true boxes
            ious = torch.tensor([calculate_iou(pb, tb) for tb in true_boxes_class])
            # If IoU of the predicted box with any of true boxes is greater than threshold, it is a true positive
            if ious.max() >= iou_threshold:
                tp += 1

        # Precision = TP / (TP + FP)
        precision = tp / predicted_boxes_class.shape[0]
        precisions.append(precision)

    return precisions


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

def display_img(img_data, fig, axes):
    '''
    Display the image in the axes.
    '''
    print(img_data.shape)
    if len(img_data.shape) == 4 and img_data.shape[1] == 3:
        img_data = img_data.permute(0, 2, 3, 1)
    print(img_data.shape)
    img_data = img_data.numpy()
    img = img_data[0]

    axes[0].imshow(img)
    axes[0].set_title('Image 1')

    img = img_data[1]
    axes[1].imshow(img)
    axes[1].set_title('Image 2')

    return fig, axes
def display_bbox(gt_bboxes, fig, ax, classes=None, line_width=2, color='r'):
    '''
    Display the bounding boxes in the axes.
    '''
    # # get the image
    # img_data = img_data_all[0]
    # img_data = img_data.permute(1, 2, 0)
    # img_data = img_data.numpy()

    # # display the image
    # ax.imshow(img_data)
    ax.set_title('Image 1')

    # display the bounding boxes
    for i, bbox in enumerate(gt_bboxes):
        if bbox[0] == -1:
            break
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=line_width, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        if classes:
            ax.text(xmin, ymin, classes[i], color='white', fontsize=12)

    return fig, ax

def display_grid(anchor_pts_x, anchor_pts_y, fig, ax, anc_point_coord=None):
    '''
    Display the grid mapping in the axes.
    '''
    # # get the image
    # img_data = img_data_all[0]
    # img_data = img_data.permute(1, 2, 0)
    # img_data = img_data.numpy()
    #
    # # display the image
    # ax.imshow(img_data)
    # ax.set_title('Image 1')

    if anc_point_coord is not None:
        c = 'w'
    else:
        c = 'b'

    # display the grid mapping
    for x in anchor_pts_x:
        for y in anchor_pts_y:
            #add small plus marker at each point
            if (x, y) == anc_point_coord:
                ax.plot(x, y, 'r+', color='r')
            else:
                ax.plot(x, y, 'r+', color=c)


    return fig, ax

def project_bboxes(bboxes, width_scale_factor, height_scale_factor, mode='a2p'):
    assert mode in ['a2p', 'p2a']

    batch_size = bboxes.size(dim=0)
    proj_bboxes = bboxes.clone().reshape(batch_size, -1, 4)
    invalid_bbox_mask = (proj_bboxes == -1) # indicating padded bboxes

    if mode == 'a2p':
        # activation map to pixel image
        proj_bboxes[:, :, [0, 2]] *= width_scale_factor
        proj_bboxes[:, :, [1, 3]] *= height_scale_factor
    else:
        # pixel image to activation map
        proj_bboxes[:, :, [0, 2]] /= width_scale_factor
        proj_bboxes[:, :, [1, 3]] /= height_scale_factor

    proj_bboxes.masked_fill_(invalid_bbox_mask, -1) # fill padded bboxes back with -1
    proj_bboxes.resize_as_(bboxes)

    return proj_bboxes


def gen_anc_centers(out_size):
    out_h, out_w = out_size

    anc_pts_x = torch.arange(0, out_w) + 0.5
    anc_pts_y = torch.arange(0, out_h) + 0.5

    return anc_pts_x, anc_pts_y

def gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, out_size, device='cpu'):
    n_anc_boxes = len(anc_scales) * len(anc_ratios)
    anc_base = torch.zeros(1, anc_pts_x.size(dim=0) \
                              , anc_pts_y.size(dim=0), n_anc_boxes, 4) # shape - [1, Hmap, Wmap, n_anchor_boxes, 4]

    for ix, xc in enumerate(anc_pts_x):
        for jx, yc in enumerate(anc_pts_y):
            anc_boxes = torch.zeros((n_anc_boxes, 4))
            c = 0
            for i, scale in enumerate(anc_scales):
                for j, ratio in enumerate(anc_ratios):
                    w = scale * ratio
                    h = scale

                    xmin = xc - w / 2
                    ymin = yc - h / 2
                    xmax = xc + w / 2
                    ymax = yc + h / 2

                    anc_boxes[c, :] = torch.Tensor([xmin, ymin, xmax, ymax])
                    c += 1

            anc_base[:, ix, jx, :] = ops.clip_boxes_to_image(anc_boxes, size=out_size)

    return anc_base


def get_iou_mat(batch_size, anc_boxes_all, gt_bboxes_all):
    # flatten anchor boxes
    anc_boxes_flat = anc_boxes_all.reshape(batch_size, -1, 4)
    # get total anchor boxes for a single image
    tot_anc_boxes = anc_boxes_flat.size(dim=1)

    # create a placeholder to compute IoUs amongst the boxes
    ious_mat = torch.zeros((batch_size, tot_anc_boxes, gt_bboxes_all.size(dim=1)))

    # put ious_mat on the same device as the anchor boxes
    ious_mat = ious_mat.to(anc_boxes_all.device)

    # compute IoU of the anc boxes with the gt boxes for all the images
    for i in range(batch_size):
        gt_bboxes = gt_bboxes_all[i]
        anc_boxes = anc_boxes_flat[i]
        ious_mat[i, :] = ops.box_iou(anc_boxes, gt_bboxes)

    return ious_mat

def calc_gt_offsets(pos_anc_coords, gt_bbox_mapping):
    pos_anc_coords = ops.box_convert(pos_anc_coords, in_fmt='xyxy', out_fmt='cxcywh')
    gt_bbox_mapping = ops.box_convert(gt_bbox_mapping, in_fmt='xyxy', out_fmt='cxcywh')

    gt_cx, gt_cy, gt_w, gt_h = gt_bbox_mapping[:, 0], gt_bbox_mapping[:, 1], gt_bbox_mapping[:, 2], gt_bbox_mapping[:, 3]
    anc_cx, anc_cy, anc_w, anc_h = pos_anc_coords[:, 0], pos_anc_coords[:, 1], pos_anc_coords[:, 2], pos_anc_coords[:, 3]

    tx_ = (gt_cx - anc_cx)/anc_w
    ty_ = (gt_cy - anc_cy)/anc_h
    tw_ = torch.log(gt_w / anc_w)
    th_ = torch.log(gt_h / anc_h)

    return torch.stack([tx_, ty_, tw_, th_], dim=-1)


def get_req_anchors(anc_boxes_all, gt_bboxes_all, gt_classes_all, pos_thresh=0.7, neg_thresh=0.2):
    '''
    Prepare necessary data required for training

    Input
    ------
    anc_boxes_all - torch.Tensor of shape (B, w_amap, h_amap, n_anchor_boxes, 4)
        all anchor boxes for a batch of images
    gt_bboxes_all - torch.Tensor of shape (B, max_objects, 4)
        padded ground truth boxes for a batch of images
    gt_classes_all - torch.Tensor of shape (B, max_objects)
        padded ground truth classes for a batch of images

    Returns
    ---------
    positive_anc_ind -  torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    negative_anc_ind - torch.Tensor of shape (n_pos,)
        flattened positive indices for all the images in the batch
    GT_conf_scores - torch.Tensor of shape (n_pos,), IoU scores of +ve anchors
    GT_offsets -  torch.Tensor of shape (n_pos, 4),
        offsets between +ve anchors and their corresponding ground truth boxes
    GT_class_pos - torch.Tensor of shape (n_pos,)
        mapped classes of +ve anchors
    positive_anc_coords - (n_pos, 4) coords of +ve anchors (for visualization)
    negative_anc_coords - (n_pos, 4) coords of -ve anchors (for visualization)
    positive_anc_ind_sep - list of indices to keep track of +ve anchors
    '''
    # get the size and shape parameters
    B, w_amap, h_amap, A, _ = anc_boxes_all.shape
    N = gt_bboxes_all.shape[1] # max number of groundtruth bboxes in a batch

    # get total number of anchor boxes in a single image
    tot_anc_boxes = A * w_amap * h_amap

    # get the iou matrix which contains iou of every anchor box
    # against all the groundtruth bboxes in an image
    iou_mat = get_iou_mat(B, anc_boxes_all, gt_bboxes_all)

    # for every groundtruth bbox in an image, find the iou
    # with the anchor box which it overlaps the most
    max_iou_per_gt_box, _ = iou_mat.max(dim=1, keepdim=True)

    # get positive anchor boxes

    # condition 1: the anchor box with the max iou for every gt bbox
    positive_anc_mask = torch.logical_and(iou_mat == max_iou_per_gt_box, max_iou_per_gt_box > 0)
    # condition 2: anchor boxes with iou above a threshold with any of the gt bboxes
    positive_anc_mask = torch.logical_or(positive_anc_mask, iou_mat > pos_thresh)

    positive_anc_ind_sep = torch.where(positive_anc_mask)[0] # get separate indices in the batch
    # combine all the batches and get the idxs of the +ve anchor boxes
    positive_anc_mask = positive_anc_mask.flatten(start_dim=0, end_dim=1)
    positive_anc_ind = torch.where(positive_anc_mask)[0]

    # for every anchor box, get the iou and the idx of the
    # gt bbox it overlaps with the most
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_anc = max_iou_per_anc.flatten(start_dim=0, end_dim=1)

    # get iou scores of the +ve anchor boxes
    GT_conf_scores = max_iou_per_anc[positive_anc_ind]

    # get gt classes of the +ve anchor boxes

    # expand gt classes to map against every anchor box
    gt_classes_expand = gt_classes_all.view(B, 1, N).expand(B, tot_anc_boxes, N)
    # for every anchor box, consider only the class of the gt bbox it overlaps with the most
    GT_class = torch.gather(gt_classes_expand, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1)
    # combine all the batches and get the mapped classes of the +ve anchor boxes
    GT_class = GT_class.flatten(start_dim=0, end_dim=1)
    GT_class_pos = GT_class[positive_anc_ind]

    # get gt bbox coordinates of the +ve anchor boxes

    # expand all the gt bboxes to map against every anchor box
    gt_bboxes_expand = gt_bboxes_all.view(B, 1, N, 4).expand(B, tot_anc_boxes, N, 4)
    # for every anchor box, consider only the coordinates of the gt bbox it overlaps with the most
    GT_bboxes = torch.gather(gt_bboxes_expand, -2, max_iou_per_anc_ind.reshape(B, tot_anc_boxes, 1, 1).repeat(1, 1, 1, 4))
    # combine all the batches and get the mapped gt bbox coordinates of the +ve anchor boxes
    GT_bboxes = GT_bboxes.flatten(start_dim=0, end_dim=2)
    GT_bboxes_pos = GT_bboxes[positive_anc_ind]

    # get coordinates of +ve anc boxes
    anc_boxes_flat = anc_boxes_all.flatten(start_dim=0, end_dim=-2) # flatten all the anchor boxes
    positive_anc_coords = anc_boxes_flat[positive_anc_ind]

    # calculate gt offsets
    GT_offsets = calc_gt_offsets(positive_anc_coords, GT_bboxes_pos)

    # get -ve anchors

    # condition: select the anchor boxes with max iou less than the threshold
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.where(negative_anc_mask)[0]
    # sample -ve samples to match the +ve samples
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (positive_anc_ind.shape[0],))]
    negative_anc_coords = anc_boxes_flat[negative_anc_ind]

    return positive_anc_ind, negative_anc_ind, GT_conf_scores, GT_offsets, GT_class_pos, \
         positive_anc_coords, negative_anc_coords, positive_anc_ind_sep


def generate_proposals(anchors, offsets):
    # change format of the anchor boxes from 'xyxy' to 'cxcywh'
    anchors = ops.box_convert(anchors, in_fmt='xyxy', out_fmt='cxcywh')

    # apply offsets to anchors to create proposals
    proposals_ = torch.zeros_like(anchors)
    proposals_[:, 0] = anchors[:, 0] + offsets[:, 0] * anchors[:, 2]
    proposals_[:, 1] = anchors[:, 1] + offsets[:, 1] * anchors[:, 3]
    proposals_[:, 2] = anchors[:, 2] * torch.exp(offsets[:, 2])
    proposals_[:, 3] = anchors[:, 3] * torch.exp(offsets[:, 3])

    # change format of proposals back from 'cxcywh' to 'xyxy'
    proposals = ops.box_convert(proposals_, in_fmt='cxcywh', out_fmt='xyxy')

    return proposals

def calc_cls_loss(conf_scores_pos, conf_scores_neg, batch_size):
    #calculate classification loss for +ve and -ve samples per batch
    loss = F.binary_cross_entropy_with_logits(conf_scores_pos, torch.ones_like(conf_scores_pos), reduction='sum') + \
        F.binary_cross_entropy_with_logits(conf_scores_neg, torch.zeros_like(conf_scores_neg), reduction='sum')
    loss = loss / batch_size



    return loss

def calc_bbox_reg_loss(GT_offsets, offsets_pos, batch_size):
    # calculate the loss for the +ve samples
    loss = F.smooth_l1_loss(offsets_pos, GT_offsets, reduction='sum')

    # normalize the loss
    loss = loss / batch_size

    return loss

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        req_layers = list(model.children())[:8]
        self.backbone = nn.Sequential(*req_layers)

    def forward(self, x):
        # get the feature maps from the backbone
        feature_map = self.backbone(x)

        # print(f"X: {x.shape}")
        # print(f"feature_map: {feature_map.shape}")
        # # add a channel dimension to the feature map
        # feature_map = feature_map.unsqueeze(0)
        #
        # print(f"Unsqueezed feature_map: {feature_map.shape}")

        return feature_map


import torch


def calculate_iou(box1, box2):
    '''Calculate Intersection over Union of two bounding boxes'''

    # box = (x1, y1, x2, y2)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (box1_area + box2_area - intersection)




