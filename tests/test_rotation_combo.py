import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from rotations import rotate_box

def test_x_rotations():
  rotations = 1
  box_size = 2
  pre_box = [[0, 1, 0, 15]] 

  