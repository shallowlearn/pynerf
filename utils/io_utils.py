'''
A module containing utilities to read from and write into different file formats
Author: Bharath Comandur
'''
import numpy as np
import os
import sys
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

def doesfileexist(f, stop=False):
    '''
    A function that checks if a file exists. If stop is True and it
    does not exist, it will throw error and stop program
    Parameters
    ----------
    f : Path to file
    stop : If stop is True and it does not exist, it will throw error and stop program
           If stop is False, it returns if file exists or not

    Returns
    -------

    check: If stop is True and it does not exist, it will throw error and stop program
           If stop is False, it returns if file exists or not
    '''
    check = os.path.isfile(f)
    if not check and stop:
        print("ERROR: {} not found".format(f))
        sys.exit(1)
    else:
        return check

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='')
#     #parser.add_argument('-', '--', required=False, type=str, default=None, help='')
#     #parser.add_argument('-', '--', action='store_true', help='')
#
#     args = parser.parse_args()
#     main(args)