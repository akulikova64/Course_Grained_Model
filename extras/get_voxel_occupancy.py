import numpy as np
import csv
import sys
import math
# This script counts the voxel occupancy for any boxed and binned data structure.

### Parameters/ data
box_size = 1 # (box dimensions) ex: 9x9x9
voxel_size = 9
center_index = math.ceil(box_size/2) - 1
output_path = "../data/output/box_analysis/box_size_" + str(box_size) + "_voxel_size_" + str(voxel_size) + ".csv"
input_path = "../data/input/boxes_s" + str(box_size) + "_" + str(voxel_size) + "A/boxes_train.npy"


#===================================================================================================
# Main
#===================================================================================================
# loading data (preboxes)
preboxes = np.load(input_path, allow_pickle = True).tolist()
# prebox structure: [x_index (0-(box_size-1)), y_index, z_index, aa_index (0-19)] EX: [3,0,8,19]
# list of voxel occupanccy counts for each box.
total_voxels = box_size**3 # total voxels per box
voxel_counts = np.zeros((box_size, box_size, box_size, len(preboxes)))
for i, prebox in enumerate(preboxes):
  for ind_set in prebox:
    x, y, z = ind_set[0], ind_set[1], ind_set[2]
    voxel_counts[x, y, z, i] += 1

# get the maximum occupancy number:
max_occ = 0
for box in range(0, len(preboxes)):
  for x in range(0, box_size):
    for y in range(0, box_size):
      for z in range(0, box_size):
        voxel_occ = voxel_counts[x, y, z, box]
        if int(voxel_occ) > max_occ:
          max_occ = voxel_occ

# get the center voxel count/density
center_count = []
for box in range(0, len(preboxes)):
  center_count.append(voxel_counts[center_index, center_index, center_index, box])

# now we can count up the occupancy types (empty, 1aa, 2aa, 3aa, 4 or more aa)
count_summary = np.zeros((int(max_occ) + 1, len(preboxes)))
'''
structure of count_summary:
occupancy:    [0,  1,  2, 3, 4 ... max_occ] 
prebox_1:     [50, 40, 6, 7, 0 ] <- how many voxels with 0aa, 1aa, 3aa... in prebox_1
prebox_2:     [80, 20, 4, 8, 0]
prebox_3:     [10, 70, 6, 5, 0]
'''

for box in range(0, len(preboxes)):
  for x in range(0, box_size):
    for y in range(0, box_size):
      for z in range(0, box_size):
        voxel_occ = voxel_counts[x, y, z, box]
        count_summary[int(voxel_occ), box] += 1

# creating a CSV file:
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)

    # adding header to CSV 
    header = []
    header.append("box_size")
    header.append("voxel_size")
    header.append("total_voxels")
    header.append("center_density")
    for i in range(0, int(max_occ) + 1):
      header.append(str(i))
    writer.writerow(header) 

    # appending data to CSV
    for i in range(0, len(preboxes)):
        row = []
        row.append(box_size)
        row.append(voxel_size)
        row.append(total_voxels)
        row.append(center_count[i])
        for j in range(0, int(max_occ) + 1):
          row.append(int(count_summary[j][i]))
        writer.writerow(row)

print("Finished making csv.")
  


