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
    # print(max_class_index)
    # print(max_class_score)
    thresholded_detections = torch.cat((thresholded_detections[:, 0:5], max_class_index.unsqueeze(1).float(), max_class_score.unsqueeze(1)), 1)
    # print(thresholded_detections)

    unique_classes = max_class_index.unique()
    for unique_class_index in unique_classes:
        detections_of_class = thresholded_detections[thresholded_detections[:, 5] == unique_class_index.float()]
        # print("Class: ", unique_class_index.item())
        # print(detections_of_class.shape)
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
        # print("Sorted:", sorted_detections_of_class.shape)

def center_coord_to_diagonals(center_coords_with_dimension):
    """
    Get coordinates for upper left and bottom right diagonals given center coordinates and dimensions of rectangle
    :param center_coords_with_dimension: array_like
                                         (center_coord_x, center_coord_y, rectangle_width, rectangle_height)
    :return: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    center_x, center_y, width, height = center_coords_with_dimension
    top_left_x, bottom_right_x = center_x - width / 2, center_x + width / 2
    top_left_y, bottom_right_y = center_y - height / 2, center_y + height / 2
    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

def compute_iou(detection_1, detection_2):
    """
    Compute the Intersection/Union of two detection frames
    :param detection_1: array_like
                        Float, shape (7)
                        coord_of_center, y_coord_of_center, width, height, obj_score, class, class_conf
    :param detection_1: array_like
                        Float, shape (7)
                        coord_of_center, y_coord_of_center, width, height, obj_score, class, class_conf
    :return: Float - Intersecion/Union
    """
    top_left_x_1, top_left_y_1, bottom_right_x_1, bottom_right_y_1 = center_coord_to_diagonals(detection_1[:4]);
    top_left_x_2, top_left_y_2, bottom_right_x_2, bottom_right_y_2 = center_coord_to_diagonals(detection_2[:4]);

    intersection_top_x = torch.max(top_left_x_1, top_left_x_2)
    intersection_top_y = torch.max(top_left_y_1, top_left_y_2)
    intersection_bottom_x = torch.min(bottom_right_x_1, bottom_right_x_2)
    intersection_bottom_y = torch.min(bottom_right_y_1, bottom_right_y_2)
    intersection_area = torch.clamp((intersection_bottom_x - intersection_top_x + 1) * (intersection_bottom_y - intersection_top_y + 1), min=0)

    detection_1_area = (bottom_right_x_1 - top_left_x_1 + 1) * (bottom_right_y_1 - top_left_y_1 + 1)
    detection_2_area = (bottom_right_x_2 - top_left_x_2 + 1) * (bottom_right_y_2 - top_left_y_2 + 1)
    union_area = detection_1_area + detection_2_area - intersection_area

    iou = intersection_area / union_area

    return torch.tensor([iou])