import numpy as np

predictions = np.load("../output/predictions_model26.npy", allow_pickle = True)
for box in predictions:
  print(box)
