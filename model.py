import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConvNet(nn.Module):
    def __init__(self, input_shape):
        super(BasicConvNet,self).__init__()
        self.window_size, self.n_features = input_shape

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,5), stride=1)
        self.relu = F.relu

        last_layer_shape =[self.window_size - 2*2 - 4, self.n_features - 2*2 - 4]
        self.fc1 = nn.Linear(last_layer_shape[0]*last_layer_shape[1]*32, 32)
        self.fc2 = nn.Linear(32, 1) 

    
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # [# data, width, height, channels] => [data, c, w, h]
        
        conv1_out = self.conv1(x)
        conv1_out = self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out = self.relu(conv3_out)

        flattened = torch.flatten(conv3_out, 1)
        fc1_out = self.fc1(flattened)
        fc1_out = self.relu(fc1_out)
        
        fc2_out = self.fc2(fc1_out)
        fc2_out = self.relu(fc2_out)

        return fc2_out

