import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt




class CNNFitness(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNFitness, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=3)
        self.conv3 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=5)

        return 