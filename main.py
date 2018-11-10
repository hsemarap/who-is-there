import utils
import torch
import pprint
import cv2
import os
from torch.autograd import Variable
from yolo import *

model = Yolo()
model.load_weights()
coco_classes = utils.get_coco_classes()
image_list = [constants.TEST_IMAGE]

def detect_image(image_np):
    target_dimension = int(model.meta["height"])
    processed_img = utils.process_image(image_np, target_dimension)
    image_dimension = torch.FloatTensor([image_np.shape[1], image_np.shape[0]])
    scaling_factor = torch.min(target_dimension / image_dimension)
    image_var = Variable(processed_img)
    # 416 * 416 * (1/(8*8) + 1/(16*16) + 1/(32*32) )*3
    output = model(image_var, False)

    # print("output", output.shape)
    thresholded_output = utils.object_thresholding(output[0])
    # print("Thresholded", thresholded_output.shape)
    # print(output[0])
    true_output = utils.non_max_suppression(thresholded_output)
    # print("True output", true_output.shape)
    if (true_output.size(0) > 0):
        # Offset for padded image
        vertical_offset = (target_dimension - scaling_factor * image_dimension[0].item()) / 2
        horizontal_offset = (target_dimension - scaling_factor * image_dimension[1].item()) / 2
        for output_box in true_output:
            rect_coords = utils.center_coord_to_diagonals(output_box[:4])
            rect_coords = torch.FloatTensor(rect_coords)
            # transform box detection w.r.t. boundaries of the padded image
            rect_coords[[0, 2]] -= vertical_offset
            rect_coords[[1, 3]] -= horizontal_offset
            rect_coords /= scaling_factor
            # Clamp to actual image's boundaries
            rect_coords[[0, 2]] = torch.clamp(rect_coords[[0, 2]], 0.0, image_dimension[0])
            rect_coords[[1, 3]] = torch.clamp(rect_coords[[1, 3]], 0.0, image_dimension[1])

            # print(image_np.shape)
            class_label = coco_classes[output_box[5].int()]
            image_np = utils.draw_box(rect_coords, image_np, class_label)
    return image_np

def perform_detection(image_list):
    images_np, image_names = [], []
    for image_path in image_list:
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        images_np.append(image)
        image_names.append(image_name)

    for i, (image_np, image_name) in enumerate(zip(images_np, image_names)):
        output_image = detect_image(image_np)
        output_name = "output/det_" + image_name
        cv2.imwrite(output_name, output_image)

def cam(idx=0):
    video_capture = cv2.VideoCapture(idx)
    video_capture.set(3, 720)
    video_capture.set(4, 720)

    print("Quit by pressing 'x'")

    while True:
        isRead, read_frame = video_capture.read()

        if isRead:
            output_frame = detect_image(read_frame)
            cv2.imshow('Webcam', output_frame)
        else:
            print("Error")
            break

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('x'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

perform_detection(image_list)
# cam()