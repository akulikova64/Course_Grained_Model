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

def append_text(line):
  file = open("prints_gen.txt", "a+")
  file.write(line + "\n")
  file.close()

def append_val(line):
  file = open("prints_val.txt", "a+")
  file.write(line + "\n")
  file.close()
  

# generator for training data
def dataGenerator_rot(pre_boxes, center_aa_list, batch_size, rotations, BLUR, center_prob, box_size):
  """ generates data for training in batches with rotations and shuffling """

  file = open("prints_gen.txt", "w+")
  file.close()
  print("training generator started")
  
  batch_fraction = int(batch_size/rotations)
  prebox_len = len(pre_boxes)
  box_list = []
  center_list = []

  while True:
    zip_lists = list(zip(pre_boxes, center_aa_list)) # first, zipping the two list to synchronize them by index
    random.shuffle(zip_lists)
    pre_boxes, center_aa_list = list(zip(*zip_lists)) # unzipping after shuffling

    for i in range(prebox_len):
      rotated_preboxes = rotation_combo(pre_boxes[i], rotations, box_size)

      for rotated_prebox in rotated_preboxes:
        center_list.append(center_aa_list[i])
        append_text("center: " + str(center_aa_list[i]) + " " + str(i))
        if BLUR == False:
          box_list.append(make_one_box(rotated_prebox, box_size))
        else:
          box_list.append(make_blurred_box(rotated_prebox, center_prob, box_size))
        
      if (i + 1) == prebox_len: 
        yield np.asarray(box_list), to_categorical(center_list, 20)
        append_text("last box: " + str(i))
        box_list = []
        center_list = []
        continue

      if (i + 1) % batch_fraction == 0:
        yield np.asarray(box_list), to_categorical(center_list, 20)
        append_text("box: " + str(i))
        box_list = []
        center_list = []
    

# generator for testing and validation data
def dataGenerator_no_rot(pre_boxes, center_aa_list, batch_size, BLUR, center_prob, box_size):
  """ data generator for testing and validation in batches (does not rotate data)"""
  
  file = open("prints_val.txt", "w+")
  file.close()

  print("no rotations generator started")

  prebox_len = len(pre_boxes)
  box_list = []
  center_list = []
  while True:
    for i in range(prebox_len):
      if BLUR == False:
        box = make_one_box(pre_boxes[i], box_size)
      else:
        box = make_blurred_box(pre_boxes[i], center_prob, box_size)

      box_list.append(box)
      center_list.append(center_aa_list[i])

      if (i + 1) == prebox_len: 
        yield np.asarray(box_list), to_categorical(center_list, 20)
        append_val("last validation box: " + str(i))
        box_list = []
        center_list = []
        continue

      if (i + 1) % batch_size == 0 :
        yield np.asarray(box_list), to_categorical(center_list, 20)
        append_val("validation box: " + str(i))
        box_list = []
        center_list = []


# generator for prediction
def predict_dataGenerator(pre_boxes, batch_size, BLUR, center_prob, box_size):
  """ data generator for testing and validation in batches (does not rotate data)"""
  print("prediction generator started")

  box_list = []
  prebox_len = len(pre_boxes)

  while True:
    for i in range(prebox_len):
      if BLUR == False:
        box = make_one_box(pre_boxes[i], box_size)
      else:
        box = make_blurred_box(pre_boxes[i], center_prob, box_size)

      box_list.append(box)

      if (i + 1) == prebox_len:
        yield np.asarray(box_list)
        box_list = []
        continue

      if (i + 1) % batch_size == 0:
        yield np.asarray(box_list)
        box_list = []

