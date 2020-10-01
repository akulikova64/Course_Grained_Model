import random
import numpy as np
from rotations import rotation_combo
from box_maker import make_one_box
from box_maker import make_blurred_box

try:
  from keras.utils import to_categorical
except ImportError:
  from tensorflow.keras.utils import to_categorical

# data generators:

# generator for training data
def dataGenerator_rot(pre_boxes, center_aa_list, batch_size, rotations, BLUR, center_prob, box_size):
  """ generates data for training in batches with rotations and shuffling """

  print("training generator started")
  zip_lists = list(zip(pre_boxes, center_aa_list)) # first, zipping the two list to synchronize them by index
  random.shuffle(zip_lists)
  pre_boxes, center_aa_list = list(zip(*zip_lists)) # unzipping after shuffling
  
  batch_fraction = int(batch_size/rotations)
  box_list = []
  center_list = []
  for i in range(0, len(pre_boxes)):
    rotated_preboxes = rotation_combo(pre_boxes[i], rotations, box_size)

    for rotated_prebox in rotated_preboxes:
      center_list.append(center_aa_list[i])
      if BLUR == False:
        box_list.append(make_one_box(rotated_prebox, box_size))
      else:
        box_list.append(make_blurred_box(rotated_prebox, center_prob, box_size))
      
    if i + 1 == len(pre_boxes): 
      yield np.asarray(box_list), to_categorical(center_list, 20)
      break

    if i + 1 % batch_fraction == 0 :
      yield np.asarray(box_list), to_categorical(center_list, 20)
      box_list = []
      center_list = []

# generator for testing and validation data
def dataGenerator_no_rot(pre_boxes, center_aa_list, batch_size, BLUR, center_prob, box_size):
  """ data generator for testing and validation in batches (does not rotate data)"""
  
  print("validation/test generator started")

  box_list = []
  center_list = []
  for i in range(0, len(pre_boxes)):

    if BLUR == False:
      box = make_one_box(pre_boxes[i], box_size)
    else:
      box = make_blurred_box(pre_boxes[i], center_prob, box_size)

    box_list.append(box)
    center_list.append(center_aa_list[i])

    if i + 1 == len(pre_boxes): 
      yield np.asarray(box_list), to_categorical(center_list, 20)
      break

    if i + 1 % batch_size == 0 :
      yield np.asarray(box_list), to_categorical(center_list, 20)
      box_list = []
      center_list = []

  '''
  while True:
    for i in range(0, len(pre_boxes) - batch_size, batch_size):
      box_list = []
      center_list = []
      for j in range(i, i + batch_size): 
        if BLUR == False:
          box = make_one_box(pre_boxes[j], box_size)
        else:
          box = make_blurred_box(pre_boxes[j], center_prob, box_size)
        
        box_list.append(box)
        center_list.append(center_aa_list[j])

      yield np.asarray(box_list), to_categorical(center_list, 20)'''

# generator for prediction
def predict_dataGenerator(pre_boxes, batch_size, BLUR, center_prob, box_size):
  """ data generator for testing and validation in batches (does not rotate data)"""
  print("prediction generator started")

  i = 1
  box_list = []
  len_boxes = len(pre_boxes)

  while True:
    if BLUR == False:
      box = make_one_box(pre_boxes[i-1], box_size)
    else:
      box = make_blurred_box(pre_boxes[i-1], center_prob, box_size)

    box_list.append(box)

    if i == len_boxes:
      yield np.asarray(box_list)
      break

    if i % batch_size == 0:
      yield np.asarray(box_list)
      box_list = []

    i += 1

