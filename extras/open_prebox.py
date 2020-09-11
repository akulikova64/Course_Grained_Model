import numpy as np
import sys

preboxes = np.load("../boxes_38/boxes_1b4t.npy", allow_pickle=True)
print(type(preboxes))
print(preboxes[0])
sys.exit()

preboxes = np.load("../validation/boxes_normalized.npy", allow_pickle=True)
#print(preboxes[0])
#print(type(np.array(preboxes)))
print(type(preboxes[0]))
sys.exit()
print(preboxes)



preboxes = np.load("../boxes/boxes_train.npy", allow_pickle=True)
print(len(preboxes))

centers = np.load("../boxes/centers_train.npy", allow_pickle=True)
print(centers)
print(len(centers))

preboxes = np.load("../validation/boxes_normalized.npy", allow_pickle=True)
#print(preboxes)
print(len(preboxes))

centers = np.load("../validation/centers_normalized.npy", allow_pickle=True)
print(centers)
print(len(centers))

preboxes = np.load("../testing/boxes_normalized.npy", allow_pickle=True)
#print(preboxes)
print(len(preboxes))

centers = np.load("../testing/centers_normalized.npy", allow_pickle=True)
print(centers)
print(len(centers))



