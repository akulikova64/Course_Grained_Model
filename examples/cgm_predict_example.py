try:
  from keras.models import load_model
  from keras.optimizers import Adam
  from keras.callbacks import ModelCheckpoint
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import ModelCheckpoint

from prebox_maker import process_residue
from prebox_maker import get_residues_df
from predictor import predict

import numpy as np
import os
import math
import time
from datetime import datetime
def timestamp():
  return str(datetime.now().time())

# this code is for runnning cnn in prediction mode

def get_voxel_index(coord, center):
  global DELTA, VOXEL_SIZE
  minimum = center - DELTA
  # we need to set the min to zero
  index = math.floor((coord - minimum)/VOXEL_SIZE)
  return index

def fill_box_for_predict(i_center, SCcenter_coords, seq):
  aa_dict = {'H':0, 'E':1, 'D':2,  'R':3, 'K':4, 'S':5, 'T':6, 'N':7, 'Q':8, 'A':9, \
             'V':10, 'L':11, 'I':12, 'M':13, 'F':14, 'Y':15, 'W':16, 'P':17, 'G':18, 'C':19}

  box = []
  center = SCcenter_coords[i_center]
    
  for i, coords in enumerate(SCcenter_coords):
      xyz_ind = []
      for j in range (0, 3):
        xyz_ind.append(get_voxel_index(coords[j], center[j]))

      if min(xyz_ind) >= 0 and max(xyz_ind) < 9:
        aa = seq[i]
        aa_ind = aa_dict[aa]

        if i_center == i:
          continue
        else:
          box.append([xyz_ind[0], xyz_ind[1], xyz_ind[2], aa_ind]) #[x, y, z, aa index]
  
  return box

def get_all_preboxes(SCcenter_coords, seq):
  # creating, filling and appending 4D box arrays to list
  box_list = []
  for center_aa_index in range(0, len(seq)):
    new_box = fill_box_for_predict(center_aa_index, SCcenter_coords, seq)
    box_list.append(new_box)
  
  return box_list

#--------main--------------
# variables:
BOX_SIZE = 63 # length in A of a box dimension
DELTA = BOX_SIZE/2
VOXEL_SIZE = 7 # bin length in A 
VOXELS = int(BOX_SIZE/VOXEL_SIZE)
BATCH_SIZE = 20
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel

# data paths
path = "../PDB/"
destination_path = "../predictions/"

fileList = os.listdir(path)

for file in fileList:
  pdb_id = file[0:4] # ex:"1b4t"
  try:
    residues_df = get_residues_df(pdb_id)
  except TypeError:
    print(pdb_id)
    continue
    
  ### retrieving amino acid coordinates and sequence
  SCcenter_coords = residues_df["SCcenter_coords"].values.tolist()
  seq = residues_df["amino_acid"].values.tolist()
  print("Found the aa side chain center coords and seqence:", timestamp(), "\n")

  ### making preboxes for the entire protein (all residues)
  box_list = get_all_preboxes(SCcenter_coords, seq)
  print("Loaded all the preboxes:", timestamp(), "\n")

  ### loading trained model
  model = load_model("../output/model_30_20200909-234817.h5")
  print("Loaded model:", timestamp(), "\n")

  ### making and saving predictions 
  predictions = predict(model, box_list, BATCH_SIZE, BLUR, center_prob, VOXELS)
  print("predictions made:", timestamp(), "\n")
  np.save(destination_path + "predictions_" + str(pdb_id) + ".npy", predictions)
  print("Finished predicting:", timestamp(), "\n")
