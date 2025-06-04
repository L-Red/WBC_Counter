"""image_splitting.py: This file defines various function to the splitting and reconstruction of large images. It includes:
- splitting a large image into a grid of smaller ones
- creating feature pyramids by sampling at different resoltions
- running inference on different resolutions
- combining boxes detected at different scales and neighbouring image patches
"""

import math

import cv2
import torch
import numpy as np
from torchvision.ops import nms, box_iou
import torch.nn.functional as F


def get_idx2name():
    idx2name = {
        0: 'LY',
        1: 'RBC',
        2: 'PLT',
        3: 'EO',
        4: 'MO',
        5: 'BNE',
        6: 'SEN',
        7: 'BA',
    }
    return idx2name

def get_name2idx():
    idx2name = get_idx2name()
    name2idx = {v: k for k, v in idx2name.items()}
    return name2idx

def generate_sliding_windows(image, stepSize, windowSize, boxes = None, labels = None):
    list_windows = []
    cnt = 0
    split_boxes = []
    split_labels = []
    split_images = []
    im_height = image.size(dim=1)
    im_width = image.size(dim=2)
    train = boxes is not None and labels is not None
    for x in range(0, im_width, stepSize):
        col = []
        for y in range(0, im_height, stepSize):
            # Create an empty image filled with zeros (black)
            img = torch.zeros((3, windowSize, windowSize))

            # Compute the boundaries of the region to copy from the original image
            x1, y1 = x, y
            x2, y2 = min(x + windowSize, im_width), min(y + windowSize, im_height)

            # Compute the size of the region to copy
            copy_width, copy_height = x2 - x1, y2 - y1

            # Copy the region from the original image
            img[:, :copy_height, :copy_width] = image[:, y1:y2, x1:x2]
            if img.size(dim=0) != 3:
                continue
            col.append(img)
            if train:
                indices = ((boxes[:,0] >= x1) & (boxes[:,2] <= x2) & (boxes[:,1] >= y1) & (boxes[:,3] <= y2)).nonzero(as_tuple=True)[0]
                split_box = boxes[indices].clone().to(dtype=torch.float32)
                split_box[:, [0, 2]] -= x1
                split_box[:, [1, 3]] -= y1
                split_box[:, [0, 2]] /= (x2 - x1)
                split_box[:, [1, 3]] /= (y2 - y1)
                split_boxes.append(split_box.to(dtype=torch.long))
                split_labels.append(labels[indices].clone())
            cnt += 1
        split_images.append(col)
    if len(split_images[0]) == 0:
        img = torch.zeros((3, windowSize, windowSize))
        img[:, :im_height, :im_width] = image
        split_images = [[img]]
    if train:
        return split_images, split_boxes, split_labels
    else:
        return split_images, None, None

def split_image_and_boxes(image, boxes=None, labels=None, split_size=720, step_size=360):

    return generate_sliding_windows(image, stepSize=step_size, windowSize=split_size, boxes=boxes, labels=labels)

def merge_overlapping_boxes(boxes, labels, scores, threshold):
    if boxes.size(0) == 0:
        return boxes, labels, scores

    ious = box_iou(boxes, boxes)  # Compute IoU
    ious.fill_diagonal_(0)  # Set self-overlaps to 0

    while True:
        max_iou = ious.max()
        if max_iou < threshold:
            break  # No more overlaps to merge

        max_index = torch.where(ious == max_iou)
        if max_index[0].numel() == 0:
            break

        box_a = boxes[max_index[0][0]]
        box_b = boxes[max_index[1][0]]

        # Compute average box (you can customize this as needed)
        new_box = (box_a + box_b) / 2

        # Choose the score and label of the box with the highest score
        if scores[max_index[0][0]] > scores[max_index[1][0]]:
            new_score = scores[max_index[0][0]]
            new_label = labels[max_index[0][0]]
        else:
            new_score = scores[max_index[1][0]]
            new_label = labels[max_index[1][0]]

        # Remove the original boxes, scores, and labels and add the new ones
        boxes = torch.cat([boxes[:max_index[0][0]], boxes[max_index[0][0]+1:]])
        boxes = torch.cat([boxes[:max_index[1][0]], boxes[max_index[1][0]+1:]])
        boxes = torch.cat([boxes, new_box.unsqueeze(0)])

        scores = torch.cat([scores[:max_index[0][0]], scores[max_index[0][0]+1:]])
        scores = torch.cat([scores[:max_index[1][0]], scores[max_index[1][0]+1:]])
        scores = torch.cat([scores, new_score.unsqueeze(0)])

        labels = torch.cat([labels[:max_index[0][0]], labels[max_index[0][0]+1:]])
        labels = torch.cat([labels[:max_index[1][0]], labels[max_index[1][0]+1:]])
        labels = torch.cat([labels, new_label.unsqueeze(0)])

        # Recompute IoUs
        ious = box_iou(boxes, boxes)
        ious.fill_diagonal_(0)

    return boxes, labels, scores



def  reconstruct_boxes(split_boxes, split_labels, split_scores, split_size=720, step_size=360, img_size=None, iou_threshold=0.5):
    h, w = img_size
    # num_splits_h = math.ceil((h-split_size) / step_size) + 1
    # num_splits_w = math.ceil((h-split_size) / step_size) + 1

    all_boxes = []
    all_labels = []
    all_scores = []

    idx = 0
    for j, splt_bxs in enumerate(split_boxes):
        for i, bxs in enumerate(splt_bxs):
            start_h, end_h = i*step_size, min(i*step_size + split_size, h)
            start_w, end_w = j*step_size, min(j*step_size + split_size, w)

            # adjust boxes
            boxes = bxs.clone()
            boxes[:, [0, 2]] += start_w  # adjust for the original image's coordinate system
            boxes[:, [1, 3]] += start_h

            boxes = boxes.clamp(min=0)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(max=w)  # assuming w is the width of the image
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(max=h)  # as

            scores = split_scores[j][i]
            labels = split_labels[j][i]

            # # apply NMS per fragment
            # nms_indices = nms(boxes, scores, iou_threshold)
            # boxes = boxes[nms_indices]
            # scores = scores[nms_indices]
            # labels = labels[nms_indices]

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)

            idx += 1

    # concatenate all boxes, labels, and scores
    all_boxes = torch.cat(all_boxes, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_scores = torch.cat(all_scores, dim=0)

    # all_boxes, all_labels, all_scores = merge_overlapping_boxes(all_boxes, all_labels, all_scores, threshold=0.5)

    # # apply NMS globally
    # nms_indices = nms(all_boxes, all_scores, iou_threshold)
    # all_boxes = all_boxes[nms_indices]
    # all_labels = all_labels[nms_indices]
    # all_scores = all_scores[nms_indices]




    return all_boxes, all_labels, all_scores


def yolo_to_rcnn(batch_outputs, do_nms=True, iou_threshold=0.5, remove_rbc=False):
    dicts = []
    name2idx = get_name2idx()
    for pred in batch_outputs.pred:
        if remove_rbc:
            pred = pred[pred[:, 5] != name2idx['RBC']]
        d = {'boxes': pred[:, :4], 'labels': pred[:, 5].long(), 'scores': pred[:, 4]}
        # apply NMS
        if do_nms:
            nms_indices = nms(d['boxes'], d['scores'], iou_threshold)
            d['boxes'] = d['boxes'][nms_indices]
            d['labels'] = d['labels'][nms_indices]
            d['scores'] = d['scores'][nms_indices]
        dicts.append(d)
    return dicts

def remove_smallest_boxes(boxes, labels, scores, keep_fraction=0.3):
    # Compute areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort boxes, labels, and scores by area (in descending order)
    sorted_indices = areas.argsort(descending=True)
    boxes = boxes[sorted_indices]
    labels = labels[sorted_indices]
    scores = scores[sorted_indices]

    # Remove the smallest 70% of boxes
    cutoff = int(keep_fraction * boxes.shape[0])
    boxes = boxes[:cutoff]
    labels = labels[:cutoff]
    scores = scores[:cutoff]

    return boxes, labels, scores



def split_inference_reconstruct(model, images, split_size=720, batch_size=2, model_name='yolo', update_progress_callback=None, remove_bottom=True):
    # Initialize lists to store results
    all_results = []

    # check if images are in a batch (tuple)
    if not isinstance(images, tuple):
        images = (images,)

    model.eval()
    for idx, img in enumerate(images):
        step_size = split_size // 2 // 2.1

        # Create the image pyramid
        pyramid = create_image_pyramid(img, [1.0, 0.5, 0.25])

        all_boxes = []
        all_labels = []
        all_scores = []

        for scaled_img, scale_factor in pyramid:


            step_size = int(step_size * scale_factor)
            split_size = int(split_size * scale_factor)

            # Split images
            split_images, _, _ = split_image_and_boxes(scaled_img, split_size=split_size, step_size=step_size)

            print(f'Running inference on {len(split_images)} splits')
            all_outputs = []
            # Run inference on the batch of splits in groups of batch_size
            batch_size = 8
            ws = len(split_images)
            hs = len(split_images[0])
            i = 0
            j = 0
            split_boxes = []
            split_labels = []
            split_scores = []
            for i in range(len(split_images)):
                splt = split_images[i]
                splt_bxs = []
                splt_lbls = []
                splt_scrs = []
                for j in range(0, len(splt), batch_size):
                    batch = splt[j:min(j + batch_size, hs)]
                    print(f'Running inference on batch {i*hs+j+1}/{ws*hs}')
                    with torch.no_grad():
                        if model_name == 'rcnn':
                            batch_outputs = model(tuple(batch))  # Model takes a tuple of images
                        elif model_name == 'yolo':
                            # split images to numpy
                            print(f'max value: {batch[0].max()}')
                            print(f'min value: {batch[0].min()}')
                            if batch[0].max() <= 1.0:
                                model_input = [si.permute(1, 2, 0).numpy() * 255 for si in batch]
                            else:
                                model_input = [si.permute(1, 2, 0).numpy() for si in batch]
                            # run inference
                            batch_outputs = model(model_input)
                            batch_outputs = yolo_to_rcnn(batch_outputs, remove_rbc=False, do_nms=True, iou_threshold=0.5)
                            print(f'Example bbox: {batch_outputs[0]["boxes"]}')

                    all_outputs.extend(batch_outputs)
                    # Update progress bar
                    if update_progress_callback is not None:
                        # Update the progress: calculate percentage and emit signal
                        progress = (i + 1) / len(split_images) * 100
                        update_progress_callback(progress)

                    # Extract split boxes, labels, and scores
                    splt_bxs.extend([output['boxes'] for output in batch_outputs])
                    splt_lbls.extend([output['labels'] for output in batch_outputs])
                    splt_scrs.extend([output['scores'] for output in batch_outputs])
                split_boxes.append(splt_bxs)
                split_labels.append(splt_lbls)
                split_scores.append(splt_scrs)

            # Adjust box coordinates based on scale_factor
            new_split_boxes = []
            for sb in split_boxes:
                new_split_boxes.append([adjust_boxes_for_scale(box, 1 / scale_factor) for box in sb])
            split_boxes = new_split_boxes
            print('Done adjusting boxes')
            print(f'Example box: {split_boxes[0][0]}')

            step_size = int(step_size / scale_factor)
            split_size = int(split_size / scale_factor)

            # Reconstruct boxes, labels, and scores
            if len(split_boxes) > 1 or len(split_boxes[0]) > 1:
                print('Reconstructing boxes')
                boxes, labels, scores = reconstruct_boxes(
                    split_boxes,
                    split_labels,
                    split_scores,
                    split_size=int(split_size),
                    step_size=int(step_size),
                    img_size=scaled_img.shape[-2:],
                    iou_threshold=0.3
                )
            else:
                boxes = split_boxes[0][0]
                labels = split_labels[0][0]
                scores = split_scores[0][0]
            print('Done reconstructing boxes')

            all_boxes.append(boxes)
            all_labels.append(labels)
            all_scores.append(scores)

        # Concatenate boxes, labels, and scores from all scales
        boxes = torch.cat(all_boxes)
        labels = torch.cat(all_labels)
        scores = torch.cat(all_scores)

        # adjust boxes for original image size
        # boxes = adjust_boxes_for_scale(boxes, 1 / scale_factor)

        # apply NMS globally
        nms_indices = nms(boxes, scores, 0.3)
        boxes = boxes[nms_indices]
        labels = labels[nms_indices]
        scores = scores[nms_indices]



        # Remove the smallest 70% of boxes
        if remove_bottom:
            boxes, labels, scores = remove_smallest_boxes(boxes, labels, scores)

        # Add reconstructed results to the list
        all_results.append({
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        })


    return all_results

def classify_box(model, box, img):
    img = img.astype(np.float32) / 255
    xmin, ymin, xmax, ymax = box.detach().numpy()
    img_cropped = img[int(ymin):int(ymax), int(xmin):int(xmax), :]
    #check if picture is empty
    if img_cropped.size == 0:
        return None
    img_cropped = cv2.resize(img_cropped, (224, 224))
    img_cropped = torch.from_numpy(img_cropped).permute(2, 0, 1).unsqueeze(0)
    prediction = model(img_cropped)
    return prediction

def create_image_pyramid(image, scales):
    """
    Creates an image pyramid.

    Args:
        image (tensor): The input image.
        scales (list): A list of scales to use for the image pyramid.

    Returns:
        tuple: A list of scaled images forming the image pyramid and a list of the corresponding scale factors.
    """
    pyramid = []

    for scale in scales:
        h, w = int(image.shape[-2] * scale), int(image.shape[-1] * scale)
        # We need to unsqueeze to add a batch dimension, and then squeeze to remove it after the resize
        scaled_image = F.interpolate(image.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        pyramid.append((scaled_image, scale))

    return pyramid


def adjust_boxes_for_scale(boxes, scale_factor):
    """
    Adjust the coordinates of the bounding boxes for a given scale factor.

    Args:
        boxes (tensor): A tensor of bounding boxes.
        scale_factor (float): The scale factor.

    Returns:
        tensor: The adjusted bounding boxes.
    """
    return boxes * scale_factor

