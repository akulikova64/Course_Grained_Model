import os
import numpy as np

try:
  from keras.utils import to_categorical
except ImportError:
  from tensorflow.keras.utils import to_categorical

# box-maker

# fill a box
def make_one_box(pre_box):
  """ Makes and fills one expanded final box """
  box = np.zeros([9, 9, 9, 20]) # 4D array filled with 0

  for ind_set in pre_box:
    box[ind_set[0]][ind_set[1]][ind_set[2]][ind_set[3]] += 1

  return box

def make_blurred_box(pre_box, center_prob):
  """ Makes and fills one expanded final box with blur"""
  box = np.zeros([9, 9, 9, 20]) # 4D array filled with 0

  for ind_set in pre_box:

    # all coordinate combinations of a 3x3x3 box
    for x in range(0,3):
      for y in range(0,3):
        for z in range(0,3):

          # subtracting 1, staying same or adding 1
          x_coord = ind_set[0]+(x-1) 
          y_coord = ind_set[1]+(y-1)
          z_coord = ind_set[2]+(z-1)

           # check boundaries
          if not(x_coord >= 0 and x_coord < 9):
            continue
          if not(y_coord >= 0 and y_coord < 9):
            continue
          if not(z_coord >= 0 and z_coord < 9):
            continue

          box[x_coord][y_coord][z_coord][ind_set[3]] += get_value(x,y,z, center_prob)

  return box

def get_value(x, y, z, center_prob):
  """ Calculates the probability for the box at x, y, z (within a 3x3x3 box)"""
  center_count = (x % 2) + (y % 2) + (z % 2) # counting the number of "1" in our coordinates (centers)
  prob_unit = (1-center_prob)/14 # the probability per voxel/cube around center cube
  values = [(prob_unit/4), (prob_unit/2), prob_unit, center_prob] # corner, edge, major axis, center
  # indexes in values list correspond to the number of centers of each box type (center_count)

  return values[center_count]

# returns list of condensed boxes
def get_box_list(path): 
  """ compiles a list of preboxes from multiple files """
  fileList = os.listdir(path)
  pre_box_list = []
  center_aa_list = []

  for file in fileList:
    if "boxes" in file:
      pdb_id = file[-8:-4]

      pre_boxes = np.load(path + file, allow_pickle = True)
      for pre_box in pre_boxes:
        pre_box_list.append(pre_box)

      centers = np.load(path + "centers_" + pdb_id + ".npy", allow_pickle = True) # list of center aa's in one file
      for center in centers:
        center_aa_list.append(center)
  
  return pre_box_list, center_aa_list

# preparing testing data
def get_test_data(path_x, path_y):
  """ loads testing data into one list of expanded boxes """
  x_data_test = np.load(path_x, allow_pickle = True)
  y_data_test = np.load(path_y, allow_pickle = True)
  
  x_test = []
  for index_set  in x_data_test:
    box = make_one_box(index_set)
    x_test.append(box)

  x_test = np.asarray(x_test)
  y_test = to_categorical(y_data_test, 20)

  return x_test, y_test