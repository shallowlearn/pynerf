'''
A module containing definition of losses
Author: Bharath Comandur
'''
import numpy as np
import os
import sys
import argparse

import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class L2Loss(nn.Module):
    '''
    A simple L2Loss class that can me moved to gpu if desired
    '''

    def __init__(self, device='cuda'):
        self.device = device
        self.loss = nn.MSELoss().to(device)

    def forward(self, labels, predictions):
        return self.loss(labels, predictions)
