'''
A fully-connected Nerf network.
Author: Bharath Comandur
'''
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import yaml

class FCNerf(nn.Module):
    '''
    Implement the FC network in NERF.pdf
    Reference - "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
                Fig. 7 on Page 18
    '''
    def __init__(self, Lx = 10, Ld= 4, device=None):
        super().__init__()
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.pi = torch.tensor(np.pi, device=self.device)

        # L's for encoding x and d
        self.Lx = Lx
        self.Ld = Ld

        # in_channels = 3*2*Lx
        in_channels = 3*2*self.Lx

        # Part one consists of first five relu fc layers
        relu_fclayers_part_one = []

        # First fc layer with ReLU
        relu_fclayers_part_one += [nn.Conv1d(in_channels=in_channels, out_channels=256, kernel_size=1)]
        relu_fclayers_part_one += [nn.ReLU()]

        # Add 4 more fc layers with ReLU
        for idx in range(4):
            relu_fclayers_part_one += [nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)]
            relu_fclayers_part_one += [nn.ReLU()]
        # Place these layers in a Sequential
        self.relu_fclayers_part_one = nn.Sequential(*relu_fclayers_part_one)

        # Part two consists of next three relu fc layers
        relu_fclayers_part_two = []

        # Sixth fc layer with ReLU has input previous 256 + gamma(x) (which is of length 60)
        relu_fclayers_part_two += [nn.Conv1d(in_channels=256 + 60, out_channels=256, kernel_size=1)]
        relu_fclayers_part_two += [nn.ReLU()]

        # Add 2 more fc layers with ReLU
        for idx in range(2):
            relu_fclayers_part_two += [nn.Conv1d(in_channels=256, out_channels=256, kernel_size=1)]
            relu_fclayers_part_two += [nn.ReLU()]
        # Place these layers in a Sequential
        self.relu_fclayers_part_two = nn.Sequential(*relu_fclayers_part_two)


        # This fclayer outputs 256 feature + 1 sigma (volume density)
        self.sigma_fclayer = nn.Conv1d(in_channels=256, out_channels=257, kernel_size=1)
        # The sigma needs to be passed through a ReLU, but not the 256 dim feature vector
        self.sigma_relu = nn.ReLU()

        # This fclayer accepts the sigma_fclayer 256 + gamma(d) (length 24) as input
        gamma_d_fclayer = [nn.Conv1d(in_channels=256 + 24, out_channels=128, kernel_size=1)]
        gamma_d_fclayer += [nn.ReLU()]
        self.gamma_d_fclayer = nn.Sequential(*gamma_d_fclayer)

        # Output layer that outputs RGB
        final_fclayer = [nn.Conv1d(in_channels=128, out_channels=3, kernel_size=1)]
        final_fclayer += [nn.Sigmoid()]
        self.final_fclayer = nn.Sequential(*final_fclayer)

        return


    def gamma(self, p, L=10):
        '''
        Calculates the high frequency embedding of x. This is
        to mitigate the low-pass filtering effect of neural networks
        :param p: 3D Normalized Coordinates - Shape (B, 3)
        :return: Shape (B, 3*2*L, 1)
        '''
        gamma = torch.zeros(p.shape[0], 3*2*L, 1,  device=self.device, dtype=p.dtype)
        for power in range(L):
            '''
            Here the encoding is arranged as X,Y,Z,X,Y,Z....
            It is also possible to arrange it as X,X,X..Y,Y,Y...Z,Z,Z...
            It is not clear if this affects the performance
            '''
            gamma[:, power*6: power*6 + 3, 0] = torch.sin(torch.pow(torch.tensor(2), power) * self.pi * p)
            gamma[:, power*6 + 3: (power + 1)*6, 0] = torch.cos(torch.pow(torch.tensor(2), power) * self.pi * p)
        return gamma

    def forward(self, x, d):
        '''
        :param x: Shape (B, 3)
        :param d: Shape (B, 3)
        :return:
        '''
        # Encode x
        gamma_x = self.gamma(x, L=self.Lx)
        # gamma_x is of shape (B, 3*2*Lx, 1)

        # Run first five fcs on gamma_x
        intermediate = self.relu_fclayers_part_one(gamma_x)
        # intermediate is of shape (B, 256, 1)

        # Concatenate gamma_x with intermediate along dim = 1 (channel axis)
        intermediate = torch.cat((intermediate, gamma_x), dim=1)

        # Run remaining three fcs on intermediate
        intermediate = self.relu_fclayers_part_two(intermediate)

        # Run sigma_fclayer on intermediate
        intermediate = self.sigma_fclayer(intermediate)

        # Sigma is in index 0 and we pass it through relu
        sigma = self.sigma_relu(intermediate[:, 0:1])

        # Encode d
        gamma_d = self.gamma(d, L=self.Ld)
        # gamma_d is of shape (B, 3*2*Ld, 1)

        # Concatenate the last 256 features of intermediate with gamma_d along dim = 1
        intermediate = torch.cat((intermediate[:, 1:], gamma_d), dim=1)

        # intermediate is of shape (B, 256 + 24, 1)
        # Pass intermediate through gamma_d_fclayer
        intermediate = self.gamma_d_fclayer(intermediate)

        # intermediate is of shape (B, 128, 1)
        # Pass intermediate through output
        output = self.final_fclayer(intermediate)

        return output, sigma

def main(args):

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #parser.add_argument('-', '--', required=False, type=str, default=None, help='')
    #parser.add_argument('-', '--', action='store_true', help='')
 
    args = parser.parse_args()
    main(args)