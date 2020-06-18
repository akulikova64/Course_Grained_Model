import os
import sys
import numpy as np
# This program subsets data lists into training, testing and validation 
N_TEST = 400
PDB_LIST = []

def subset_for_testing():
  global N_TEST
  global PDB_LIST

  box_path = './boxes/'
  test_path = './testing/'

  for i, line in enumerate(PDB_LIST):
    if i >= N_TEST:
      break
    pdb_id = line[0:4]
    PDB_LIST.remove(line)
  
    os.rename(box_path + "boxes_" + pdb_id + ".npy", test_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", test_path + "centers_" + pdb_id + ".npy")

# make sure that the pdbs are removed from the training 
def subset_for_validation(box_path, valid_path):
  global N_TEST
  global PDB_LIST

  for i, line in enumerate(PDB_LIST):
    if i >= N_TEST:
      break
    pdb_id = line[0:4]
    PDB_LIST.remove(line)

    # the function rename() moves files
    os.rename(box_path + "boxes_" + pdb_id + ".npy", valid_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", valid_path + "centers_" + pdb_id + ".npy")

def normalize_count(): # for the test and validation datasets, we make each aa center appear equal times.
  global PDB_LIST
  aa_count = np.zeros(20) # indices in this list encode the 20 aa. 
  
  for pdb in PDB_LIST:
    centers = np.load("./testing/centers_" + pdb + ".npy", allow_pickle = True)
    for aa in centers:
      aa_count[aa] += 1

  min_count = int(min(aa_count)) # Count of the least frequent aa determines the number of each aa we take.
  '''
  print(min_count)
  print(np.where(aa_count == min_count)[0])'''

  new_center_list = []
  new_box_list = []
  temp_count = np.zeros(20) #temporary aa count (needs to be <= min_count)
  for pdb in PDB_LIST:
    centers = np.load("./testing/centers_" + pdb + ".npy", allow_pickle = True)
    boxes = np.load("./testing/boxes_" + pdb + ".npy", allow_pickle = True)
    for aa, box in zip(centers, boxes): #aa is the number encoding the amino acid
      if temp_count[aa] < min_count:
        temp_count[aa] += 1
        new_center_list.append(aa)
        new_box_list.append(box)
  
  return new_center_list, new_box_list

#---------- main ----------
pdb_id_list = open("pdb_all.list", "r")
for i, line in enumerate(pdb_id_list):
  if i >= N_TEST:
    break
  PDB_LIST.append(line[0:4])
pdb_id_list.close()

box_path = './boxes/'
valid_path = './validation/'
test_path = './testing/'
subset_for_testing(box_path, test_path)
subset_for_validation(box_path, valid_path)

new_center_list, new_box_list = normalize_count()

np.save("./testing/boxes_test.npy", np.asarray(new_box_list)) # add number to the protein and give matching number to the aa list
np.save("./testing/centers_test.npy", np.asarray(new_center_list))


