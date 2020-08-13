import os
import sys
import numpy as np
# This program subsets data lists into training, testing and validation 
N_TEST = 4
PDB_LIST = []
TEST_LIST = []

def subset_for_testing(box_path, test_path):
  global N_TEST
  global PDB_LIST

  for i, line in enumerate(PDB_LIST):
    if i >= N_TEST:
      break
    pdb_id = line[0:4]
  
    os.rename(box_path + "boxes_" + pdb_id + ".npy", test_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", test_path + "centers_" + pdb_id + ".npy")
  
  for i in range(0, N_TEST):
    TEST_LIST.append(PDB_LIST[i])

  del PDB_LIST[0:4]
    
# make sure that the pdbs are removed from the training 
def subset_for_validation(box_path, valid_path):
  global N_TEST
  global PDB_LIST

  for i, line in enumerate(PDB_LIST):
    if i >= N_TEST:
      break
    pdb_id = line[0:4]

    # the function rename() moves files
    os.rename(box_path + "boxes_" + pdb_id + ".npy", valid_path + "boxes_" + pdb_id + ".npy")
    os.rename(box_path + "centers_" + pdb_id + ".npy", valid_path + "centers_" + pdb_id + ".npy")

  del PDB_LIST[0:4]
  
def normalize_count(): # for the test and validation datasets, we make each aa center appear equal times.
  global TEST_LIST
  aa_count = np.zeros(20) # indices in this list encode the 20 aa. 
  
  for pdb in TEST_LIST:
    centers = np.load("../testing_2/centers_" + pdb + ".npy", allow_pickle = True)
    for aa in centers:
      aa_count[aa] += 1

  min_count = int(min(aa_count)) # Count of the least frequent aa determines the number of each aa we take.
  '''
  print(min_count)
  print(np.where(aa_count == min_count)[0])'''

  new_center_list = []
  new_box_list = []
  temp_count = np.zeros(20) #temporary aa count (needs to be <= min_count)
  for pdb in TEST_LIST:
    centers = np.load("../testing_2/centers_" + pdb + ".npy", allow_pickle = True)
    boxes = np.load("../testing_2/boxes_" + pdb + ".npy", allow_pickle = True)
    for aa, box in zip(centers, boxes): #aa is the number encoding the amino acid
      if temp_count[aa] < min_count:
        temp_count[aa] += 1
        new_center_list.append(aa)
        new_box_list.append(box)
  
  return new_center_list, new_box_list

# ---------- main ----------
pdb_id_list = open("../pdb_all.list", "r")

'''for i, line in enumerate(pdb_id_list):
  if i >= N_TEST:
    break
  PDB_LIST.append(line[0:4])
pdb_id_list.close()'''

for line in pdb_id_list:
  PDB_LIST.append(line[0:4])
pdb_id_list.close()

box_path = '../boxes_2/'
valid_path = '../validation_2/'
test_path = '../testing_2/'
subset_for_testing(box_path, test_path)
subset_for_validation(box_path, valid_path)

new_center_list, new_box_list = normalize_count()

np.save("../testing_2/boxes_test.npy", np.asarray(new_box_list)) # add number to the protein and give matching number to the aa list
np.save("../testing_2/centers_test.npy", np.asarray(new_center_list))


