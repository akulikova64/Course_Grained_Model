import sys
import os
from box_maker import get_value

# parameters to get_value: x, y, z, center_prob 
# x, y, z refer to indexes of small blurred box (3x3x3)

def test_center_value():
  assert get_value(1, 1, 1, 0.44) == 0.44
  assert get_value(1, 1, 1, 0) == 0
  assert get_value(1, 1, 1, 1) == 1

def test_major_axis_value():
  assert get_value(1, 1, 0, 0.44) == 0.04
  assert get_value(0, 1, 1, 0.44) == 0.04
  assert get_value(1, 0, 1, 0.44) == 0.04
  assert get_value(2, 1, 1, 0.44) == 0.04
  assert get_value(1, 2, 1, 0.44) == 0.04
  assert get_value(1, 1, 2, 0.44) == 0.04 

def test_edge_value():
  assert get_value(0, 1, 0, 0.44) == 0.02
  assert get_value(0, 2, 1, 0.44) == 0.02
  assert get_value(1, 0, 2, 0.44) == 0.02
  assert get_value(2, 1, 0, 0.44) == 0.02

def test_corner_value():
  assert get_value(0, 0, 0, 0.44) == 0.01
  assert get_value(2, 2, 2, 0.44) == 0.01
  assert get_value(2, 0, 0, 0.44) == 0.01
  assert get_value(2, 0, 2, 0.44) == 0.01