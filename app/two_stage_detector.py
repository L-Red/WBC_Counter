"""
two_stage_detector.py: In this file, we define our two stage detector model. 
It can either be instatiated with a YOLO or a Faster RCNN backend. It outputs bounding boxes, classes and confidence scores to the caller.
"""

import torch
import torchvision
from torch.nn.functional import softmax

from app import image_splitting


class TwoStageDetector(object):
    def __init__(self, yolo_path, resnet_path, model_name='yolo'):
        if model_name == 'rcnn':
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
            # load weights
            self.model.load_state_dict(
                torch.load('../torch_rcnn_try/runs/run7_noweights/model_run4_balance_v2_80_noweights.pt',
                           map_location=device))
        elif model_name == 'yolo':
            # Load the saved file
            self.model = torch.hub.load('../yolov5', 'custom', path=yolo_path,
                                   source='local')  # local model
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.idx2name = image_splitting.get_idx2name()
        self.model_name = model_name

        self.resnet = torchvision.models.resnet50()
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, len(self.idx2name.keys()))
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=self.device))

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)

    def detect(self, tensor_image, verbose=False, remove_bottom=True, small_image=False):
        if not isinstance(tensor_image, tuple):
            tensor_image = (tensor_image,)
        if self.model_name == 'rcnn':
            results = image_splitting.split_inference_reconstruct(self.model, tensor_image, split_size=600, model_name=self.model_name, remove_bottom=remove_bottom)  # Assumes this function exists in image_splitting.py
        elif self.model_name == 'yolo':
            if small_image:
                img = [ti.permute(1, 2, 0).numpy() for ti in tensor_image]
                results = self.model(img)
                results = image_splitting.yolo_to_rcnn(results)
            else:
                results = image_splitting.split_inference_reconstruct(self.model, tensor_image, split_size=1200, model_name=self.model_name, remove_bottom=remove_bottom)  # Assumes this function exists in image_splitting.py
            # Draw the bounding boxes and labels on the image
        outputs = []
        if verbose:
            print(f'Running inference on boxes...')
        for ti, result in zip(tensor_image, results):
            out_boxes = []
            out_labels = []
            out_scores = []
            out_yolo_labels = []
            out_yolo_scores = []
            print(f'Amount of boxes: {len(result["boxes"])}')
            for box, label, score, i in zip(result['boxes'], result['labels'], result['scores'],
                                            range(0, len(result['boxes']))):
                if verbose:
                    print(f'Box {i} of {len(result["boxes"])}')
                yolo_label = self.idx2name[label.item()]
                yolo_score = score.item()

                # check that box doesn't have side == 0
                if box[0] == box[2] or box[1] == box[3]:
                    print(f'Box {i} has side == 0, skipping.')
                    continue


                prediction = image_splitting.classify_box(self.resnet, box, ti.permute(1, 2, 0).numpy())
                if prediction is None:
                    print(f'Box {i} has no prediction, skipping.')
                    continue


                prediction = softmax(prediction)
                conf, label = torch.max(prediction, 1)
                label = self.idx2name[label.item()]
                box = [int(b.item()) for b in box]
                confidences = conf[0].item()

                # cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # cv2.putText(self.image, str(label), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
                if label == 'RBC' and yolo_label == 'RBC':
                    print(f'Box {i} has both labels RBC, skipping.')
                    continue
                out_boxes.append(box)
                out_labels.append(label)
                out_scores.append(confidences)
                out_yolo_labels.append(yolo_label)
                out_yolo_scores.append(yolo_score)
            output = {
                'boxes': out_boxes,
                'labels': out_labels,
                'scores': out_scores,
                'yolo_labels': out_yolo_labels,
                'yolo_scores': out_yolo_scores
            }
            outputs.append(output)
        if verbose:
            print(f'Finished inference on boxes.')
        return outputs
