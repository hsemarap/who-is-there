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