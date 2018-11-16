import constants
import cv2
import os
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


def resize_image_fixed_aspect_ratio(image, input_dimensions):
    """
    Resize image without altering it's aspect ratio, by padding with pixels valued 128
    :param image: Input image
    :param input_dimensions: array_like
            Target dimensions of the image [height, width]
    :return:
    """
    image_width, image_height = image.shape[1], image.shape[0]
    width, height = input_dimensions
    ratio = min(width / image_width, height / image_height)
    new_width = int(image_width * ratio)
    new_height = int(image_height * ratio)
    new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    #create new image filled with value 128
    resized_image = np.full((height, width, 3), 128)
    embed_image_top, embed_image_left = (height - new_height) // 2, (width - new_width) // 2
    embed_image_bottom, embed_image_right = embed_image_top + new_height, embed_image_left + new_width
    #Embed the resized image into padded image
    resized_image[embed_image_top:embed_image_bottom, embed_image_left:embed_image_right, :] = new_image

    return resized_image

def process_image(image, target_height):
    resized_image = resize_image_fixed_aspect_ratio(image, (target_height, target_height))
    processed_image = resized_image[:, :, ::-1].transpose((2, 0, 1))
    processed_image = processed_image[np.newaxis, :, :, :] / 255.0
    processed_image = torch.from_numpy(processed_image).float()
    return processed_image

def get_coco_classes():
    f = open(constants.COCO_CLASSES, "r")
    coco_classes = f.read().splitlines()
    return coco_classes

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
    if thresholded_detections.size(0) == 0:
        return thresholded_detections
    max_class_score, max_class_index = torch.max(thresholded_detections[:, 5:5 + constants.NUM_CLASSES], 1)
    # print(max_class_index)
    # print(max_class_score)
    thresholded_detections = torch.cat((thresholded_detections[:, 0:5], max_class_index.unsqueeze(1).float(), max_class_score.unsqueeze(1)), 1)
    # print(thresholded_detections)
    true_output_list = []
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
        true_output_list.append(sorted_detections_of_class)
    true_output = torch.cat(true_output_list)
    # print("True:")
    # print(true_output)
    return true_output

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

def draw_box(rectangle_coords, image, label):
    c1 = tuple(rectangle_coords[:2].int())
    c2 = tuple(rectangle_coords[2:4].int())
    color = (255, 0, 0)
    cv2.rectangle(image, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(image, c1, c2,color, -1)
    cv2.putText(image, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return image