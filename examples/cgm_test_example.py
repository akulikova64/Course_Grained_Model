from datetime import datetime
import numpy as np

try:
  from keras.models import load_model
except ImportError:
  from tensorflow.keras.models import load_model

def timestamp():
  return str(datetime.now().time())

# returns testing results
def get_testing_results(model, box_size, batch_size, BLUR, center_prob, x_test, y_test):
  """ testing the trained model """

  # figure out how to use the test_val_generator for the testing data
  '''
  x_test = []
  for index_set in x_test_preboxes: # why are we not going through boxes first? Why do we start with index sets?
    box = make_one_box(index_set, box_size)
    x_test.append(box)

  x_test = np.asarray(x_test)
  y_test = to_categorical(y_test_centers, 20)'''

  score = model.evaluate_generator(
          generator = test_val_dataGenerator(x_test, y_test, batch_size, BLUR, center_prob, box_size), 
          steps = len(x_test)/batch_size
          ) 

  return score

# this code is for runnning cnn on test data. 

#-----------main----------------
# variables:
BOX_SIZE = 63 # length in A of a box dimension
BATCH_SIZE = 20
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel

# data path
testing_path = "../testing/"
model_path = "../output/model_30_20200909-234817.h5"

### loading trained model
model = load_model(model_path)
print("Loaded model:", timestamp(), "\n")

### testing
print("Finished training, loading test data:", timestamp())
x_test = np.load(testing_path + "boxes_normalized.npy", allow_pickle = True).tolist()
y_test = np.load(testing_path + "centers_normalized.npy", allow_pickle = True).tolist()
print("Finished loading test data, testing:", timestamp())
score = get_testing_results(model, BOX_SIZE, BATCH_SIZE, BLUR, center_prob, x_test, y_test)
print("Finished testing:", timestamp(), "\n")

print('--------------------------------------')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('--------------------------------------')