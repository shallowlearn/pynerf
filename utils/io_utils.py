'''
A module containing utilities to read from and write into different file formats
Author: Bharath Comandur
'''
import numpy as np
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
def read_json(f):
    '''
    A function to read a json file
    Parameters
    ----------
    f : Path to json file

    Returns
    -------

    '''
    if os.path.isfile(f):
        return json.load(open(f, 'r'))
    else:
        print("ERROR: {} NOT FOUND]n".format(f))
        sys.exit(1)
        return

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='')
#     #parser.add_argument('-', '--', required=False, type=str, default=None, help='')
#     #parser.add_argument('-', '--', action='store_true', help='')
#
#     args = parser.parse_args()
#     main(args)