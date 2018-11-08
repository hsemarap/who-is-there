import constants
import cv2
import torch
from torch.autograd import Variable
import numpy as np


def parse_config():
    with open(constants.YOLO_CONFIG_FILE) as f:
        file_content = f.read()
        config, config_dict = [], None
        for line in file_content.split('\n'):
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            if line.startswith('['):
                if config_dict is not None:
                    config.append(config_dict)
                config_dict = dict()
                config_dict['type'] = line[1:-1]
            else:
                if config_dict is None:
                    raise Exception("Got None config_dict")
                parameter, value = line.split('=')
                parameter, value = parameter.strip(), value.strip()
                config_dict[parameter] = value
        if config_dict is not None:
            config.append(config_dict)
    return config


def get_test_input():
    img = cv2.imread(constants.TEST_IMAGE)
    img = cv2.resize(img, (416, 416))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_var = Variable(img_)
    return img_var


def object_thresholding(output, threshold=0.5):
    """
    Remove detections with an objectness score below a threshold
    :param output: [10647 x 85]
    :param threshold: Float, threshold
    :return: output with rows having objectiveness score >= threshold
    """
    threshold_idx = 4
    output_thresholded_idx = output[:, threshold_idx] >= threshold
    output_thresholded = output[output_thresholded_idx]
    return output_thresholded


def non_max_suppression(thresholded_detections, iou_threshold=0.5):
    """
    Perform Non Max suppression to remove duplicate detections for same object
    :param thresholded_detections: [? x 85] = ? depends on number of detections after thresholding
    :param threshold: threshold value for IOU to consider two detections to overlap
    :return: output: [? x 7] = ? depends on NMS filtering,
            7 = 4 coordinates + object score +
    """
    max_class_score, max_class_index = torch.max(thresholded_detections[:, 5:5 + constants.NUM_CLASSES], 1)
    print(max_class_index)
    print(max_class_score)
    thresholded_detections = torch.cat((thresholded_detections[:, 0:5], max_class_index.unsqueeze(1).float(), max_class_score.unsqueeze(1)), 1)
    print(thresholded_detections)

    unique_classes = max_class_index.unique()
    for unique_class_index in unique_classes:
        detections_of_class = thresholded_detections[thresholded_detections[:, 5] == unique_class_index.float()]
        print(unique_class_index, detections_of_class)
        sorted_detections_of_class_indices = torch.sort(detections_of_class[:, 4], descending=True)[1]
        sorted_detections_of_class = detections_of_class[sorted_detections_of_class_indices]
        total_detections = sorted_detections_of_class.shape[0]
        for current_detection_index in range(total_detections):
            if current_detection_index >= sorted_detections_of_class.size(0):
                break
            iou_arr = torch.Tensor([]).float()
            current_max_detection = sorted_detections_of_class[current_detection_index, :]
            for candidate_detection in sorted_detections_of_class[current_detection_index+1:, :]:
                iou_arr = torch.cat((iou_arr, compute_iou(current_max_detection, candidate_detection)))
            candidate_detections = sorted_detections_of_class[current_detection_index + 1:, :]
            candidate_detections = candidate_detections[iou_arr < iou_threshold]
            sorted_detections_of_class = torch.cat((sorted_detections_of_class[0:current_detection_index+1, :], candidate_detections))
        print("Sorted:", sorted_detections_of_class)


def compute_iou(x, y):
    import random
    return torch.Tensor([random.random()])