from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa, three_to_one
import pandas as pd
import numpy as np
import math
import random
import os
import sys

# Global variables:
BOX_SIZE = 55 # length in A of a box dimension
DELTA = BOX_SIZE/2
VOXEL_SIZE = 5 # bin length in A ex 
VOXELS = int(BOX_SIZE/VOXEL_SIZE)
 
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
    output_dict['SCcenter_coords'] = sum(sidechain_coords)/                                      len(sidechain_coords)
    
    return output_dict

def get_residues_df(pdb_id):
  pdb_file = '../PDB/' + pdb_id + '_final_tot.pdb'
  structure = PDBParser().get_structure(pdb_id, pdb_file)
  
  atoms = structure.get_atoms()

  ###Get residue coordinates
  temp_listy = []
  cols = ['residue_number', 'amino_acid', 'chain', 'CA_coords', 'CB_coords', 'SCcenter_coords']
  for residue in structure.get_residues():
      if is_aa(residue):
          temp_dict = process_residue(residue)
          if type(temp_dict) == str:
            continue
          temp_listy.append([temp_dict[col] for col in cols])
      else:
          #print('Problem with this residue: {}'.format(residue))
          qwe = 1
          
  residues_df = pd.DataFrame(temp_listy, columns=cols)
  return residues_df

# takes in one x, y or z residue coord, the center coord (x, y or z) and returns voxel index in box
def get_voxel_index(coord, center):
    global DELTA, VOXEL_SIZE
    minimum = center - DELTA
    # we need to set the min to zero
    index = math.floor((coord - minimum)/VOXEL_SIZE)
    return index

def fill_box(i_center, SCcenter_coords, seq):
  aa_dict = {'H':0, 'E':1, 'D':2,  'R':3, 'K':4, 'S':5, 'T':6, 'N':7, 'Q':8, 'A':9, \
             'V':10, 'L':11, 'I':12, 'M':13, 'F':14, 'Y':15, 'W':16, 'P':17, 'G':18, 'C':19}

  box = []
  global center_aa_list
  center_aa_list.append(aa_dict[seq[i_center]])
  center = SCcenter_coords[i_center]
    
  for i, coords in enumerate(SCcenter_coords):
      xyz_ind = []
      for j in range (0, 3):
        xyz_ind.append(get_voxel_index(coords[j], center[j]))
      
      if min(xyz_ind) >= 0 and max(xyz_ind) < 9:
        aa = seq[i]
        aa_ind = aa_dict[aa]

        """ if box[xyz_ind[0]][xyz_ind[1]][xyz_ind[2]][aa_ind] == 1:
          print("same amino acids in one voxel, index: " + str(i) + ", aa: " + str(aa)) """ 

        if i_center == i:
          continue
        else:
          box.append([xyz_ind[0], xyz_ind[1], xyz_ind[2], aa_ind]) #[x, y, z, aa index]
  
  return box

#========================================================================================
# MAIN program below
#========================================================================================

path = "../PDB/"
fileList = os.listdir(path)
for file in fileList:
  pdb_id = file[0:4] # "1b4t"
  try:
    residues_df = get_residues_df(pdb_id)
  except TypeError:
    print(pdb_id)
    continue

  #pdb_id = sys.argv[1]
  #residues_df = get_residues_df(pdb_id)

  SCcenter_coords = residues_df["SCcenter_coords"].values.tolist()
  seq = residues_df["amino_acid"].values.tolist()

  # choosing the sample size (how many environments to be sampled from current protein)
  if len(seq) <= 200:
    sample_size = int(len(seq)/2)
  else:
    sample_size = 100

  sample = random.sample(range(0, len(residues_df)), sample_size)

  # creating, filling and appending 4D box arrays to list
  box_list = []
  center_aa_list = []

  for center_aa_index in sample:
    new_box = fill_box(center_aa_index, SCcenter_coords, seq)
    box_list.append(new_box)


  np.save("../boxes_2/boxes_" + pdb_id + ".npy", np.asarray(box_list)) # add number to the protein and give matching number to the aa list
  np.save("../boxes_2/centers_" + pdb_id + ".npy", np.asarray(center_aa_list))


# maverick2 ssh: achern64@maverick2.tacc.utexas.edu
# stampede2 ssh: achern64@stampede2.tacc.utexas.edu



