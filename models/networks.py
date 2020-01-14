import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import FeaStConv


class FeaStNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes, heads, t_inv=True):
        super(FeaStNet, self).__init__()

        self.fc0 = nn.Linear(in_channels, 16)
        self.conv1 = FeaStConv(16, 32, heads=heads, t_inv=t_inv)
        self.conv2 = FeaStConv(32, 64, heads=heads, t_inv=t_inv)
        self.conv3 = FeaStConv(64, 128, heads=heads, t_inv=t_inv)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
