import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from rotations import multiple_rotations
# testing the rotation functions in rotations.py 

# tests full rotation along z axis
def test_full_rotation():
  orig_box = np.array([[6, 7, 8, 15], [6, 8, 8, 20], [5, 7, 8, 5], [6, 4, 8, 7]])
  r1 = multiple_rotations(1, orig_box)
  r2 = multiple_rotations(3, r1)

  comparison = orig_box == r2
  equal_arrays = comparison.all()
 
  assert equal_arrays

# tests full rotation along z axis, should not pass test
def test_full_rotation_wrong():
  orig_box = np.array([[[6, 7, 8, 15], [6, 8, 8, 20], [5, 7, 8, 5], [6, 4, 8, 7]]])
  r1 = multiple_rotations(1, orig_box)
  r2 = multiple_rotations(3, r1)

  comparison = orig_box == r2
  equal_arrays = comparison.all()
  print("test2 started \n")
  print(orig_box)
  print(r1)
  print(r2)
  print(1,2,3)
 
  assert equal_arrays