import os
import numpy as np

try:
  from keras.utils import to_categorical
except ImportError:
  from tensorflow.keras.utils import to_categorical

# box-maker

# fill a box
def make_one_box(pre_box, box_size):
  """ Makes and fills one expanded final box """
  box = np.zeros([box_size, box_size, box_size, 20]) # 4D array filled with 0

  for ind_set in pre_box:
    box[ind_set[0]][ind_set[1]][ind_set[2]][ind_set[3]] += 1

  return box

def make_blurred_box(pre_box, center_prob, box_size):
  """ Makes and fills one expanded final box with blur"""
  box = np.zeros([box_size, box_size, box_size, 20]) # 4D array filled with 0

  for ind_set in pre_box:

    # all coordinate combinations of a 3x3x3 box
    for x in range(0,3):
      for y in range(0,3):
        for z in range(0,3):

          # subtracting 1, staying same or adding 1 to get all coordinates around center
          x_coord = ind_set[0]+(x-1) 
          y_coord = ind_set[1]+(y-1)
          z_coord = ind_set[2]+(z-1)

          if out_of_bound(x_coord, y_coord, z_coord, box_size): # check box boundaries (9x9x9)
            continue

          box[x_coord][y_coord][z_coord][ind_set[3]] += get_value(x,y,z, center_prob)

  return box

def out_of_bound(x_coord, y_coord, z_coord, box_size):
  # check boundaries
  if not(x_coord >= 0 and x_coord < box_size):
    return True
  if not(y_coord >= 0 and y_coord < box_size):
    return True
  if not(z_coord >= 0 and z_coord < box_size):
    return True

  return False
          
def get_value(x, y, z, center_prob):
  """ Calculates the probability for the box at x, y, z (within a 3x3x3 box)"""
  center_count = (x % 2) + (y % 2) + (z % 2) # counting the number of "1" in our coordinates (centers)
  prob_unit = (1-center_prob)/14 # the probability per voxel/cube around center cube
  values = [(prob_unit/4), (prob_unit/2), prob_unit, center_prob] # corner, edge, major axis, center
  # indexes in values list correspond to the number of centers of each box type (center_count)

  return values[center_count]
  