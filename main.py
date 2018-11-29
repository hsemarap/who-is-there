import utils
import torch
import pprint
import cv2
import os
import time
from torch.autograd import Variable
from yolo import *
import constants
import numpy as np

from Nw.Facerecognition import face_recognition_utils

model = Yolo()
model.load_weights()
coco_classes = utils.get_coco_classes()
image_list = [constants.TEST_IMAGE]
model.eval()
CUDA = torch.cuda.is_available()

if CUDA:
    model.cuda()


def detect_image(image_np):
    target_dimension = int(model.meta["height"])
    processed_img = utils.process_image(image_np, target_dimension)
    image_dimension = torch.FloatTensor([image_np.shape[1], image_np.shape[0]])
    scaling_factor = torch.min(target_dimension / image_dimension)
    if CUDA:
        processed_img = processed_img.cuda()
    image_var = Variable(processed_img)
    # 416 * 416 * (1/(8*8) + 1/(16*16) + 1/(32*32) )*3
    start = time.time()
    with torch.no_grad():
        output = model(image_var, CUDA)
    end = time.time()
    print("Total time: {}".format(end - start))

    # print("output", output.shape)
    thresholded_output = utils.object_thresholding(output[0])
    # print("Thresholded", thresholded_output.shape)
    # print(output[0])
    true_output = utils.non_max_suppression(thresholded_output)
    # print("True output", true_output.shape)
    original_image_np = np.copy(image_np)
    if true_output.size(0) > 0:
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
            print("Output Box:", output_box, "Class Label:", class_label)
            print("Rect coords:", rect_coords)
            if constants.PERFORM_FACE_DETECTION and class_label == "person":
                rc = rect_coords.int()
                person_img_np = original_image_np[rc[1]:rc[3], rc[0]:rc[2]]
                # print("person_img_np: ", person_img_np, person_img_np.shape)
                # cv2.imshow("bounded_box_img", person_img_np)
                # cv2.waitKey(0)
                face_label = face_recognition_utils.recognize_face_in_patch(person_img_np)
                if face_label is not None:
                    class_label = face_label
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


def video_detection(idx=0):
    """
    :param idx: Webcam id or the video file name
    """
    is_input_webcam = (type(idx) == int)
    video_capture = cv2.VideoCapture(idx)

    print("Quit by pressing 'x'")
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    fps = 10
    # Check if we are reading a Video File.
    if not is_input_webcam:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("Setting fps={}".format(fps))
    video_writer = cv2.VideoWriter('detection_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                                   (frame_width, frame_height))

    while True:
        isRead, read_frame = video_capture.read()

        if isRead:
            output_frame = detect_image(read_frame)
            if is_input_webcam:
                cv2.imshow('Webcam', output_frame)
            video_writer.write(output_frame)
        else:
            print("Error reading the video input source: {}".format(idx))
            break

        key_press = cv2.waitKey(1) & 0xFF
        if key_press == ord('x'):
            break

    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

#video_detection("/Users/rahul/Documents/BME-DeepLearning/project/Nw/important.mp4")
#video_detection()
perform_detection(["../IMG_2090.JPG"])