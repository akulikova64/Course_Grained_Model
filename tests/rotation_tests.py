import numpy as np
import sys
import os
sys.path.append(os.path.abspath(".."))
from rotations import multiple_rotations
# testing the rotation functions in rotations.py 

def get_box_list(path = "../boxes_38/"): 
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
def test_full_rotation():
  box_list = get_box_list()
  orig_box = box_list[5][0:5] # any box will do
  #print(orig_box)
  #print(small_array)
  #print(small_array_2)

  r1 = multiple_rotations(1, orig_box)
  #print(orig_box)
  #print(r1)
  r2 = multiple_rotations(3, r1)
  #print(orig_box)

  comparison = orig_box == r2
  equal_arrays = comparison.all()
 
  assert equal_arrays
'''
# tests full rotation along z axis, should not pass test
def test_full_rotation_wrong():
  #orig_box = np.array([[6, 7, 8, 15], [6, 8, 8, 20], [5, 7, 8, 5], [6, 4, 8, 7]])
  r1 = multiple_rotations(1, orig_box)
  r2 = multiple_rotations(3, r1)

  comparison = orig_box == r2
  equal_arrays = comparison.all()
  print("test2 started \n")
  print(orig_box)
  print(r1)
  print(r2)
  print(1,2,3)
 
  assert equal_arrays'''