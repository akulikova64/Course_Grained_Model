# this module contains rotation functions for a 9 x 9 x 9 cube. 
import math
import random

# axis is the axis across which the rotation occurs
# rot_num is the number of 90 degree rotations needed (0, 1, 2 or 3)
def rotate_box(pre_box, axis, rot_num):
  dict = {"x":[1, 2], "y":[0, 2], "z":[0, 1]} # lists the axes to be changed if rotated around key
  new_pre_box = []
  
  for ind_set in pre_box:
    a_1, a_2 = dict[axis][0], dict[axis][1]
    ind_1, ind_2 = ind_set[a_1], ind_set[a_2]
    new_set = ind_set.copy()
    a_3 = 3 - (a_1 + a_2)
    if rot_num == 1:
      new_set[a_1] = 8 - ind_2
      new_set[a_2] = ind_1
    
    if rot_num == 2:
      new_set[a_1] = 8 - ind_1
      new_set[a_2] = 8 - ind_2
      
    if rot_num == 3:
      new_set[a_1] = ind_2
      new_set[a_2] = 8 - ind_1

    new_pre_box.append(new_set)
    

  return new_pre_box

def multiple_rotations(i, pre_box):
  prebox_1 = rotate_box(pre_box, "z", i%4)

  # rotate along x or y
  rot_num = math.floor(i/4) # 0-5
  if rot_num < 4:
    prebox_2 = rotate_box(prebox_1, "y", rot_num)
  elif rot_num == 4:
    prebox_2 = rotate_box(prebox_1, "x", 1)
  elif rot_num == 5:
    prebox_2 = rotate_box(prebox_1, "x", 3)

  return prebox_2

# chooses one of 24 conformations
def rotation_combo(pre_box, rotations):
  final_preboxes = []
  rot_list = random.sample(range(0, 24), rotations)

  for i in rot_list:
    rotated_prebox = multiple_rotations(i, pre_box)
    final_preboxes.append(rotated_prebox)

  return final_preboxes

  