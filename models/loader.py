'''
A module for loading different nerf networks
Author: Bharath Comandur
'''
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

def load (name):

    if name == 'fc':
        from models.fcnerf import FCNerf
        return FCNerf

def main(args):

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='a module for loading different nerf networks')
    #parser.add_argument('-', '--', required=False, type=str, default=None, help='')
    #parser.add_argument('-', '--', action='store_true', help='')
 
    args = parser.parse_args()
    main(args)