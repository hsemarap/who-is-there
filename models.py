import utils
import constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def create_conv_layer(config_dict, module_index, prev_filters):
    if config_dict is None:
        return None
    sequential_modules = nn.Sequential()
    activation = config_dict["activation"]
    filters = int(config_dict["filters"])
    pad = int(config_dict["pad"])
    kernel_size = int(config_dict["size"])
    stride = int(config_dict["stride"])

    if "batch_normalize" in config_dict:
        batch_normalize = int(config_dict["batch_normalize"])
        bias = False
    else:
        batch_normalize = 0
        bias = True

    if pad == 1:
        padding = (kernel_size - 1) // 2
    else:
        padding = 0

    conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, padding, bias=bias)
    sequential_modules.add_module("conv_{}".format(module_index), conv)

    if batch_normalize != 0:
        batch_norm = nn.BatchNorm2d(filters)
        sequential_modules.add_module("batch_norm_{}".format(module_index), batch_norm)

    # if the activation is Linear, we don't need to do anything
    if activation == "leaky":
        leaky_activation = nn.LeakyReLU(0.1, inplace=True)
        sequential_modules.add_module("leaky_{}".format(module_index), leaky_activation)

    return sequential_modules, filters


def create_shortcut_layer(index):
    shortcut_layer = nn.Sequential()
    shortcut_layer.add_module("shortcut_{}".format(index), nn.Module())
    return shortcut_layer


def create_route_layer(config_dict, index, output_filters):
    route_layer = nn.Sequential()
    layers = config_dict["layers"].split(',')
    start = int(layers[0])
    end = int(layers[1]) if len(layers) == 2 else 0
    if start > 0:
        start = start - index
    if end > 0:
        end = end - index
    if end < 0:
        filters = output_filters[index + start] + output_filters[index + end]
    else:
        filters = output_filters[index + start]
    route_layer.add_module("route_{}".format(index), nn.Module())
    return route_layer, filters


def create_upsample_layer(config_dict, index):
    upsample_layer = nn.Sequential()
    stride = int(config_dict["stride"])
    upsample = nn.Upsample(scale_factor = stride, mode = "bilinear")
    upsample_layer.add_module("upsample_{}".format(index), upsample)
    return upsample_layer


class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors


def create_yolo_layer(config_dict, index):
    yolo_layer = nn.Sequential()
    mask_list = map(int, config_dict["mask"].split(','))
    anchors = [a for a in map(int, config_dict["anchors"].split(','))]
    anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
    anchors = [anchors[i] for i in mask_list]
    yolo = YoloLayer(anchors)
    yolo_layer.add_module("yolo_{}".format(index), yolo)
    return yolo_layer


def create_modules():
    module_list = nn.ModuleList()
    config, prev_filters, output_filters = utils.parse_config(), 3, []
    # We need to skip the "net" layer and hence start iterating  from index 1
    for index, config_dict in enumerate(config[1:]):
        if config_dict["type"] == constants.CONVOLUTIONAL_LAYER:
            conv_layer, filters = create_conv_layer(config_dict, index, prev_filters)
            module_list.append(conv_layer)
        elif config_dict["type"] == constants.SHORTCUT_LAYER:
            shortcut_layer = create_shortcut_layer(index)
            module_list.append(shortcut_layer)
            filters = prev_filters
        elif config_dict["type"] == constants.ROUTE_LAYER:
            route_layer, filters = create_route_layer(config_dict, index, output_filters)
            module_list.append(route_layer)
        elif config_dict["type"] == constants.UPSAMPLE_LAYER:
            upsample_layer = create_upsample_layer(config_dict, index)
            module_list.append(upsample_layer)
            filters = prev_filters
        elif config_dict["type"] == constants.YOLO_LAYER:
            yolo_layer = create_yolo_layer(config_dict, index)
            module_list.append(yolo_layer)
            filters = prev_filters
        output_filters.append(filters)
        prev_filters = filters
    return module_list

# print(create_modules())
