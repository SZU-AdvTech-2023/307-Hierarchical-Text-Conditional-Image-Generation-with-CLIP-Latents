from torchvision import models
import json
import os
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import torch


class BackBone(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.fc = nn.Linear(1024, 8)

    def forward(self, x):
        x = self.fc(x)
        return x

