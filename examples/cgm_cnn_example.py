# course grained cnn with rotations
from box_maker import get_box_list
from box_maker import get_test_data
from train_test import train_model
from train_test import get_testing_results
from train_test import load_model
from plot_maker import get_plots
import models

try:
  from keras.models import load_model
  from keras.optimizers import Adam
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam

import time
from datetime import datetime
def timestamp():
  return str(datetime.now().time())

#========================================================================================================
# Setting the variables, parameters and data paths/locations:
#========================================================================================================

### variables
EPOCHS = 1 # iterations through the data
ROTATIONS = 4 # number of box rotations per box
BATCH_SIZE = 20 # batch_size must be divisible by "ROTATIONS"
GPUS = 1 # max is 4 GPUs
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel
model_id = "5"
learning_rate = 0.001

### data paths/locations
training_path = "../boxes_tenth/"
validation_path = "../boxes_38/"
testing_path_x = "../testing/boxes_test.npy"
testing_path_y = "../testing/centers_test.npy"
### best models:
my_models = {"3": models.model_3, "5": models.model_5, "6": models.model_6, , "7": models.model_7, , "8": models.model_8, , "9": models.model_9, , "10": models.model_10, , "11": models.model_11}

### setting parameters for training
loss ='categorical_crossentropy'
optimizer = Adam(lr = learning_rate)
metrics = ['accuracy']

#========================================================================================================
# Training, testing and saving the cnn:
#========================================================================================================

### training and validation
print("\nStarting to load training data:", timestamp())
x_train, y_train = get_box_list(training_path) # preparing training data (boxes, centers)
print("Finished loading training data:", timestamp())
x_val, y_val = get_box_list(validation_path) # preparing validation data (boxes, centers)
print("Finished loading validation data:", timestamp())
model = my_models[model_id](GPUS)
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
print("Model compiled, starting to train:", timestamp(), "\n")
history = train_model(model, BATCH_SIZE, EPOCHS, ROTATIONS, BLUR, center_prob, x_train, y_train, x_val, y_val)

### testing
print("Finished training, loading test data:", timestamp())
x_test, y_test = get_test_data(testing_path_x, testing_path_y)
print("Finished loading test data, testing:", timestamp())
score = get_testing_results(model, BATCH_SIZE, x_test, y_test)
print("Finished testing:", timestamp(), "\n")

### saving and loading trained model
timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = "../output/model_" + model_id + "_" + timestr + ".h5"
model.save(model_name)
model = load_model(model_name)
print("Loaded model:", timestamp(), "\n")

### results
get_plots(history, model_id, BLUR, loss, optimizer, learning_rate, training_path[3:-1])
print("Making plots: ", timestamp(), "\n")
print(model.summary(), "\n")
print('Test loss:', score[0])
print('Test accuracy:', score[1])









