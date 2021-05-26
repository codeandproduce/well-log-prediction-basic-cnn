import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_conv_output_shape(shape, conv):
    height, width, _ = shape

    padding = conv.padding
    kernel_size = conv.kernel_size
    stride = conv.stride
    channels = conv.out_channels
    dilations = conv.dilation

    height = (height + 2*padding[0] - dilations[0]*(kernel_size[0]-1) - 1) / stride[0]
    height = np.floor(height+1)
    width = (width + 2*padding[1] - dilations[1]*(kernel_size[1]-1) - 1) / stride[1]
    width = np.floor(width+1)
    
    return height, width, channels


class BasicConvNet(nn.Module):
    def __init__(self, input_shape, channels, kernels, dilations):
        super(BasicConvNet,self).__init__()
        self.input_shape = input_shape
        self.input_shape.append(channels[0])

        self.channels = channels
        self.kernels = kernels
        self.dilations = dilations

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=kernels[0], stride=1, dilation=dilations[0], padding=2)
        self.conv1_shape = get_conv_output_shape(input_shape, self.conv1)
        self.conv2 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernels[1], stride=1, dilation=dilations[1])
        self.conv2_shape = get_conv_output_shape(self.conv1_shape, self.conv2)
        self.relu = F.relu

        self.last_conv_layer_shape = self.conv2_shape
        if len(channels) == 3:
            self.conv3 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernels[2], stride=1, dilation=dilations[2])
            self.conv3_shape = get_conv_output_shape(self.conv2_shape, self.conv3)
            self.relu = F.relu
            self.last_conv_layer_shape = self.conv3_shape
            
        self.fc1 = nn.Linear(np.prod(self.last_conv_layer_shape, dtype=np.int32), 32)
        self.fc2 = nn.Linear(32, 1) 

    
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [# data, width, height, channels] => [data, c, w, h]
        conv1_out = self.conv1(x)
        conv1_out = self.relu(conv1_out)
        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)

        last_cnn_layer = conv2_out
        if len(self.channels) == 3:
            conv3_out = self.conv3(conv2_out)
            conv3_out = self.relu(conv3_out)
            last_cnn_layer = conv3_out

        flattened = torch.flatten(last_cnn_layer, 1)
        fc1_out = self.fc1(flattened)
        fc1_out = self.relu(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.relu(fc2_out)

        return fc2_out

