
def get_training_list(boxes_all, boxes_frac):
  """ combines all training preboxes into a single file """

  boxes_list = []
  centers_list = []

  total = len(boxes_all)

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