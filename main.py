import utils
import torch
import pprint
from yolo import *
model = Yolo()
model.load_weights()
# test_image = utils.get_test_input()
# output = model(test_image, False)
# print(output)
# print("Final")
#416 * 416 * (1/(8*8) + 1/(16*16) + 1/(32*32) )*3
# print(output.shape)
# print(output[0])