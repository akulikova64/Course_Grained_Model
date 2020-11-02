import os
import sys
import numpy as np

# This program subsets data lists into training, testing and validation 
# and normalizes the aa class counts for both the testing and validation datasets

# To run this script, you need:
#   1) a "pdb_all.list" file with 4-chracter names of all proteins in "/boxes" folder
#   2) "/boxes" folder with generated boxes and centers from prebox_maker.py
#   3) empty "/testing" and "/validation" folders

N_TEST_VAL = 1

def move_boxes(box_path, new_path, pdb_list):
  """ moves the first "N_TEST_VAL" number of boxes from the training folder to a new folder """

  global N_TEST_VAL
  
  # moves N_TEST number of boxes to a new path
  moved_files = []
  for i, line in enumerate(pdb_list):
    if i >= N_TEST_VAL:
      break
    pdb_id = line[0:4]
  
    os.rename(box_path + "boxes_" + pdb_id + ".npy", new_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", new_path + "centers_" + pdb_id + ".npy")
  
    moved_files.append(pdb_id)

  del pdb_list[0:N_TEST_VAL]

  return moved_files, pdb_list
  
def normalize_aa_classes(list, path): 
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

  print("-------------------------------------------")
  print("Total number of boxes: ", len(boxes_list))
  print("-------------------------------------------")

  np.save(path + "boxes_train.npy", np.asarray(boxes_list)) # add number to the protein and give matching number to the aa list
  np.save(path + "centers_train.npy", np.asarray(centers_list))

# ---------- main ----------
pdb_id_list = open("../data/input/pdb_all.list", "r")
pdb_list = []

for line in pdb_id_list:
  pdb_list.append(line[0:4])
pdb_id_list.close()

box_size = "1"
voxel_size = "9"

box_path = "../data/input/boxes_s" + box_size + "_" + voxel_size + "A/"
val_path = "../data/input/validation_s" + box_size + "_" + voxel_size + "A/"
test_path = "../data/input/testing_s" + box_size + "_" + voxel_size + "A/"

# subsetting boxes to the testing or validation folders
test_list, pdb_list = move_boxes(box_path, test_path, pdb_list)
val_list, pdb_list = move_boxes(box_path, val_path, pdb_list)

# normalizing the test and validation datasets
normalize_aa_classes(test_list, test_path)
normalize_aa_classes(val_list, val_path)

# combining the remaining training preboxes into one file
get_training_list(pdb_list, box_path)

print("Finished subsetting data.")




