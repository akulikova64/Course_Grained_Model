import os
import re

path = "../boxes/"
fileList = os.listdir(path)

with open("../pdb_all.list", "w") as pdb_all_file:
  for file in fileList:
    match = re.search(r'boxes_(....).npy', file) # ex:"1b4t"
    if match:
      pbd_id = match.group(1)
      pdb_all_file.write(pbd_id + "\n")
  
