import sys
import os
sys.path.append(os.path.abspath(".."))
from box_maker import out_of_bound

# paramerters for out_of_bound: x_coord, y_coord, z_coord, box_size
def test_inside_bounds():

  assert not out_of_bound(2, 3, 5, 9)
  assert not out_of_bound(8, 1, 4, 9)
  assert not out_of_bound(10, 20, 15, 25)
  assert not out_of_bound(1, 2, 3, 4)
  
def test_outside_bounds():

  assert out_of_bound(10, 5, 6, 9)
  assert out_of_bound(5, 15, 6, 9)
  assert out_of_bound(8, 5, 12, 9)
  assert out_of_bound(20, 19, 15, 12)
  assert out_of_bound(25, 5, 12, 20)

def test_at_bounds():

  # testing -1, 0, 8 and 9
  assert out_of_bound(-1, 3, 5, 9)
  assert out_of_bound(9, 1, 4, 9)
  assert out_of_bound(9, 1, 4, 9)
  assert out_of_bound(0, 25, 4, 25)

  assert not out_of_bound(0, 1, 4, 9)
  assert not out_of_bound(1, 1, 0, 9)
  assert not out_of_bound(8, 1, 4, 9)
  assert not out_of_bound(2, 8, 4, 9) 
  

def test_negative_coords():

  assert out_of_bound(-1, 3, 5, 9)
  assert out_of_bound(8, -1, 4, 9)
  assert out_of_bound(10, 20, -1, 25)
  assert out_of_bound(-1, -2, -3, 4)
  assert out_of_bound(-1, -1, -1, 4)


