import os
import sys
import numpy as np

# This program subsets data lists into training, testing and validation 
# and normalizes the aa class counts for both the testing and validation datasets

# To run this script, you need:
#   1) a "pdb_all.list" file with 4-chracter names of all proteins in "/boxes" folder
#   2) "/boxes" folder with generated boxes and centers from prebox_maker.py
#   3) empty "/testing" and "/validation" folders

N_TEST = 4

def move_boxes(box_path, new_path, pdb_id_list):
  """ moves the first "N_TEST" number of boxes from the training folder to a new folder """

  global N_TEST
  
  # moves N_TEST number of boxes to a new path
  moved_files = []
  for i, line in enumerate(pdb_id_list):
    if i >= N_TEST:
      break
    pdb_id = line[0:4]
  
    os.rename(box_path + "boxes_" + pdb_id + ".npy", new_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", new_path + "centers_" + pdb_id + ".npy")
  
    moved_files.append(pdb_id)

  del pdb_id_list[0:4]

  return moved_files
  
def normalize_classes(list, path): 
  """ makes sure each aa appears an equal number of times for both the testing and validation sets """

  # counts the number of aa per class (20 classes)
  aa_count = np.zeros(20) # indices in this list will encode the 20 aa. 
  
  for pdb in list:
    centers = np.load(path + "centers_" + pdb + ".npy", allow_pickle = True)
    for aa in centers:
      aa_count[aa] += 1

  # finds the least frequent aa to determine the number of aa we take from each class
  min_count = int(min(aa_count)) 

  # creates two new lists of boxes and their centers with equal numbers of amino acid types
  new_centers = []
  new_boxes = []
  aa_count = np.zeros(20) # temporary aa count (needs to be <= min_count)
  
  for pdb in list:
    centers = np.load(path + "centers_" + pdb + ".npy", allow_pickle = True)
    boxes = np.load(path + "boxes_" + pdb + ".npy", allow_pickle = True)
    for aa, box in zip(centers, boxes): #aa is the number encoding the amino acid
      if aa_count[aa] < min_count:
        aa_count[aa] += 1
        new_centers.append(aa)
        new_boxes.append(box)
  
  np.save(path + "boxes_normalized.npy", np.asarray(new_boxes)) # add number to the protein and give matching number to the aa list
  np.save(path + "centers_normalized.npy", np.asarray(new_centers))

def get_training_list(list, path):
  """ combines all training preboxes into a single file """

  boxes_list = []
  centers_list = []

  for pdb in list:
    centers = np.load(path + "centers_" + pdb + ".npy", allow_pickle = True)
    boxes = np.load(path + "boxes_" + pdb + ".npy", allow_pickle = True)
    for aa, box in zip(centers, boxes): #aa is the number encoding the amino acid
      centers_list.append(aa)
      boxes_list.append(box)

  np.save(path + "boxes_train.npy", np.asarray(boxes_list)) # add number to the protein and give matching number to the aa list
  np.save(path + "centers_train.npy", np.asarray(centers_list))

# ---------- main ----------
pdb_id_list = open("../pdb_all.list", "r")

'''
for line in pdb_id_list:
  PDB_LIST.append(line[0:4])
pdb_id_list.close()'''

box_path = '../boxes/'
val_path = '../validation/'
test_path = '../testing/'

# subsetting boxes to the testing or validation folders
test_list = move_boxes(box_path, test_path, pdb_id_list)
val_list = move_boxes(box_path, val_path, pdb_id_list)

# normalizing the test and validation datasets
normalize_classes(test_list, test_path)
normalize_classes(val_list, val_path)

# combining the remaining training preboxes into one file
get_training_list(pdb_id_list, box_path)




