import torch
import torch.nn as nn
import numpy as np


class CNNFitness(nn.Module):
    def __init__(self, input_map_tensor):
        super(CNNFitness, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.input_map = torch.tensor(input_map_tensor).float().to(self.device)

        self.conv3 = nn.Conv2d(1, 1, 3, padding=1).to(self.device)
        self.conv5 = nn.Conv2d(1, 1, 5, padding=2).to(self.device)
        self.conv7 = nn.Conv2d(1, 1, 7, padding=3).to(self.device)
        self.conv9 = nn.Conv2d(1, 1, 9, padding=4).to(self.device)

        with torch.no_grad():
            for conv, k in zip([self.conv3, self.conv5, self.conv7, self.conv9], [3, 5, 7, 9]):
                conv.weight.fill_(1 / (k * k))
                if conv.bias is not None:
                    conv.bias.zero_()

    def forward(self, x):
        x = x.to(self.device) 
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x) + self.conv9(x)) / 4.0
        return out * self.input_map

    def fitness(self, corner_sensors, gene_sensors, coverage):
        return None  # 구현 예정
