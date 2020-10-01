from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
from predictor import predict
import pandas as pd
import sys
import os
import numpy as np
import math
import csv

try:
  from keras.models import load_model
  from keras.optimizers import Adam
  from keras.callbacks import ModelCheckpoint
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import ModelCheckpoint

import time
from datetime import datetime
def timestamp():
  return str(datetime.now().time())

aa_dict = {'H':0, 'E':1, 'D':2,  'R':3, 'K':4, 'S':5, 'T':6, 'N':7, 'Q':8, 'A':9, \
             'V':10, 'L':11, 'I':12, 'M':13, 'F':14, 'Y':15, 'W':16, 'P':17, 'G':18, 'C':19}

# this code is for runnning cnn in prediction mode
def process_residue(residue):
    '''
    Processes a single residue to determine the coordinates of the alpha-carbon
    and the sidechain center-of-mass. Also checks for missing atoms in a
    residue.
    '''
    output_dict = {}
    # Convert three letter amino acid to one letter
    try:
        output_dict['amino_acid'] = three_to_one(residue.resname)
    except KeyError:
        return 'Error'
    
    # Grab residue number AND any insertion site labeling (11A, 11B, etc.)
    output_dict['residue_number'] = str(residue.get_id()[1]) + residue.get_id()[2].strip()
    
    # Straightforward, grab the chain ID
    output_dict['chain'] = residue.get_full_id()[2]
    
    #Just checking correctness
    try:
        int(output_dict['residue_number'])
    except:
        return 'Error, residue number is... not an int?'
    
    # Coordinates of all sidechain atoms in this residue
    sidechain_coords = []
    atoms_seen = []
    for atom in residue:
        atoms_seen.append(atom.name)
        if atom.name == 'CA':
            # Save alpha-carbon coordinates separately
            output_dict['CA_coords'] = atom.get_coord()
            
            # If it's Glycine... call that the CB to
            if residue.resname == 'GLY':
                output_dict['CB_coords'] = atom.get_coord()
                
        if atom.name == 'CB':
            # Save beta-carbon coordinates
            output_dict['CB_coords'] = atom.get_coord()

        #Ignore the backbone and add the coordinates of all side-chain atoms
        if atom.name not in ['C', 'CA', 'O', 'N']:
            # Must be a sidechain atom...
            sidechain_coords.append(atom.get_coord())
            
    if 'CA_coords' not in output_dict or 'CB_coords' not in output_dict:
        return 'Error, this residue is missing a CA and/or a CB atom'
    
    for mainchain_atom in ['N', 'C', 'O']:
        # Warn about any missing mainchain atoms
        if mainchain_atom not in atoms_seen:
            return 'Error, this residue has a strange backbone'

    if len(sidechain_coords) == 0:
        ###Treat glycine separately. Normally CA is ignored in the side-chain atom calculations
        ###but for glycine it's all the information that we have.
        if output_dict['amino_acid'] == 'G':
            sidechain_coords.append(output_dict['CA_coords'])
        else:
            return 'Error'
        
    # Calculate side chain geometric center
    output_dict['SCcenter_coords'] = sum(sidechain_coords)/len(sidechain_coords)
    
    return output_dict

def get_residues_df(pdb_id, data_path):
  pdb_file = data_path + pdb_id + '.pdb'
  structure = PDBParser().get_structure(pdb_id, pdb_file)

  ###Get residue coordinates
  temp_listy = []
  cols = ['residue_number', 'amino_acid', 'chain', 'CA_coords', 'CB_coords', 'SCcenter_coords']
  for residue in structure.get_residues():
      if is_aa(residue):
          temp_dict = process_residue(residue)
          if type(temp_dict) == str:
            continue
          temp_listy.append([temp_dict[col] for col in cols])
          
  residues_df = pd.DataFrame(temp_listy, columns=cols)

  return residues_df

def get_voxel_index(coord, center):
  global DELTA, VOXEL_SIZE
  minimum = center - DELTA
  # we need to set the min to zero
  index = math.floor((coord - minimum)/VOXEL_SIZE)
  return index

def fill_box_for_predict(VOXELS, i_center, SCcenter_coords, seq):
  global aa_dict
  box = []
  center_coords = SCcenter_coords[i_center]
  
  for i, coords in enumerate(SCcenter_coords):
      xyz_ind = []
      for j in range (0, 3):
        xyz_ind.append(get_voxel_index(coords[j], center_coords[j]))

      if min(xyz_ind) >= 0 and max(xyz_ind) < VOXELS:
        aa = seq[i]
        aa_ind = aa_dict[aa]

        if i_center == i:
          continue
        else:
          box.append([xyz_ind[0], xyz_ind[1], xyz_ind[2], aa_ind]) #[x, y, z, aa index]
  
  return box

def get_all_preboxes(VOXELS, SCcenter_coords, seq):
  # creating, filling and appending 4D box arrays to list
  box_list = []
  for center_aa_index in range(0, len(seq)):
    new_box = fill_box_for_predict(VOXELS, center_aa_index, SCcenter_coords, seq)
    box_list.append(new_box)
  
  return box_list

def decode_predictions(predictions):
  """ Finds the winning prediction for each site in the protein. """
  global aa_dict

  winners = []
  for pred in predictions:
      max_prob = 0
      max_prob_i = 0
      for i in range(0, 20):
          if pred[i] > max_prob:
              max_prob = pred[i]
              max_prob_i = i
      for key, value in aa_dict.items():
        if value == max_prob_i:
          winners.append(key)

  return winners

def make_csv(predictions, winners, center_list, model_id, pdb_id, output_path):
  csv_file = output_path + "predictions_" + pdb_id + "_model_" + model_id + ".csv"

  with open(csv_file, 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['wt_aa', 'pred_aa', 'H', 'E', 'D', 'R', 'K', 'S', 'T', 'N', 'Q', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'P', 'G', 'C'])
      
      for i in range(0, len(winners)):
          row = []
          row.append(center_list[i])
          row.append(winners[i])
          for pred in predictions[i]:
              row.append(pred)
          writer.writerow(row)

#========================================================================================================
# Setting the variables, parameters and data paths/locations:
#========================================================================================================
### data paths/locations
data_path = "../data/input/PDB_38/"
output_path = "../data/output/predictions/"
model_path = "../data/output/training_results/model_34_run_1.h5"


### variables
model_id = "34"
BOX_SIZE = 21 # length in A of a box dimension
DELTA = BOX_SIZE/2
VOXEL_SIZE = 7 # bin length in A 
VOXELS = int(BOX_SIZE/VOXEL_SIZE)
BATCH_SIZE = 20
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel

#========================================================================================================
# Generating predictions from a trained model
#========================================================================================================

fileList = os.listdir(data_path)

for i, file in enumerate(fileList):
  pdb_id = file[0:4] # ex:"1b4t"
  try:
    residues_df = get_residues_df(pdb_id, data_path)
  except TypeError:
    print(pdb_id)
    continue

  print()
  print("(" + str(i+1) + ") Processing:" + pdb_id + ", time:", timestamp(), "\n")

  ### retrieving amino acid coordinates and sequence
  print("", timestamp(), "\n")
  SCcenter_coords = residues_df["SCcenter_coords"].values.tolist()
  seq = residues_df["amino_acid"].values.tolist()
  print("Found the aa side chain center coords and seqence:", timestamp(), "\n")

  ### making preboxes for the entire protein (all residues)
  box_list = []
  center_list = seq
  box_list = get_all_preboxes(VOXELS, SCcenter_coords, seq)
  print("Loaded all the preboxes and centers:", timestamp(), "\n")

  ### loading trained model
  model = load_model(model_path)
  print("Loaded model:", timestamp(), "\n")

  ### making predictions
  predictions = predict(model, box_list, BATCH_SIZE, BLUR, center_prob, VOXELS)
  print("Predictions made:", timestamp(), "\n")

  ### decoding predictions
  winners = decode_predictions(predictions)
  print("Predictions decoded:", timestamp(), "\n")

  '''
  print("SCcenter_coords len:", len(SCcenter_coords))
  print("seq len:", len(seq))
  print("box_list len:", len(box_list))
  print("predictions len:", len(predictions))
  print("winners len:", len(winners))
  sys.exit()'''

  ### saving data in a CSV file
  print("Starting to append", str(len(seq)), "predictions to CSV", timestamp(), "\n")
  make_csv(predictions, winners, center_list, model_id, pdb_id, output_path)
  print("Finished making CSV:", timestamp(), "\n")
  print("Completed!:", timestamp(), "\n")



