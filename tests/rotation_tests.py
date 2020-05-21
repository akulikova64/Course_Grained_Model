import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from rotations import multiple_rotations
# testing the rotation functions in rotations.py 

def get_box_list(): 
  path = "../boxes_38/"
  fileList = os.listdir(path)
  pre_box_list = []

  for file in fileList:
    if "boxes" in file:
      pdb_id = file[-8:-4]

      pre_boxes = np.load(path + file, allow_pickle = True)
      for pre_box in pre_boxes:
        pre_box_list.append(pre_box)
  
  return pre_box_list

# tests full rotation along z axis
def test_z_rotation():
  box_list = get_box_list()
  orig_box = np.asarray(box_list[5][0:5]) # any box will do

  r1 = multiple_rotations(1, orig_box)
  r2 = np.asarray(multiple_rotations(3, r1))

  comparison = orig_box == r2
  equal_arrays = comparison.all()
 
  assert equal_arrays

# tests full rotation along z axis, should not pass test
def test_z_rotation_wrong():
  box_list = get_box_list()
  orig_box = np.asarray(box_list[5][0:5]) # any box will do

  r1 = multiple_rotations(1, orig_box)
  r2 = np.asarray(multiple_rotations(2, r1))

  comparison = orig_box == r2
  equal_arrays = comparison.all()
 
  assert not equal_arrays


def test_two_rotations():
  box_list = get_box_list()
  orig_box = np.asarray(box_list[8][0:5]) # any box will do

  r1 = multiple_rotations(4, orig_box)
  r2 = multiple_rotations(17, orig_box) # 1 rot by z, 1 rot by x
  r2 = multiple_rotations(3, r2)

  comparison = np.asarray(r1) == np.asarray(r2)
  equal_arrays = comparison.all()

  assert equal_arrays