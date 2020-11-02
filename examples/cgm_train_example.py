# course grained cnn
from training_saving import train_model
from plot_maker import get_plots
from predictor import get_val_predictions
import models
import numpy as np
import time
import sys
import re
import os

try:
  from keras.models import load_model
  from keras.optimizers import Adam
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam

from datetime import datetime
def timestamp():
  return str(datetime.now().time())

def get_current_run(model_id, output_path):
  """ checks output folder for the previous run number and adds one """

  path = output_path + "/"
  fileList = os.listdir(path)

  current_run = 1
  for file in fileList:
    match_1 = re.search(r'model_' + model_id + '_', file)
    if match_1:
      match_2 = re.search(r'run_([0-9]+).h5', file)
      if match_2:
        run = int(match_2.group(1))
        if run >= current_run:
          current_run = run + 1

  return current_run

def compile_model(model_id, run, my_models, GPUS, BOX_SIZE, loss, optimizer, metrics, output_path):
  """ loads previous model or compiles a new model """

  last_model = output_path + "/model_" + model_id + "_run_" + str(run-1) + ".h5"
  try:
    model = load_model(last_model)
    print("Loaded last model:", last_model)
  except:
    model = my_models[model_id](GPUS, BOX_SIZE)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    print("No previous model found. Training as Run 1 (from zero) :")

  print("Model loaded/compiled:", timestamp(), "\n")

  return model

def load_data(training_path, validation_path):
  """ loads training and validation data """

  print("\nStarting to load training data:", timestamp())
  x_train = np.load(training_path + "boxes_train.npy", allow_pickle = True).tolist()
  y_train = np.load(training_path + "centers_train.npy", allow_pickle = True).tolist()
  print("Finished loading training data:", timestamp())
  x_val = np.load(validation_path + "boxes_normalized.npy", allow_pickle = True).tolist()
  y_val = np.load(validation_path + "centers_normalized.npy", allow_pickle = True).tolist()
  print("Finished loading validation data:", timestamp())

  return x_train[0:8], y_train[0:8], x_val[0:8], y_val[0:8]


#========================================================================================================
# Setting the variables, parameters and data paths/locations:
#========================================================================================================
### data paths/locations
training_path = "../data/input/boxes/"
validation_path = "../data/input/validation/"
output_path = "../data/output/training_results"

### variables
EPOCHS = 2 # iterations through the data
BOX_SIZE = 9 # number of bins in the x,y or z directions
ROTATIONS = 1 # number of box rotations per box
BATCH_SIZE = 2 # batch_size must be divisible by "ROTATIONS"
GPUS = 1 # max is 4 GPUs
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel
model_id = "31"
learning_rate = 0.0001
run = get_current_run(model_id, output_path)

### best models:
my_models = {"31": models.model_31, "32": models.model_32, "33": models.model_33, "34": models.model_34}

### setting parameters for training
loss ='categorical_crossentropy'
optimizer = Adam(lr = learning_rate)
metrics = ['accuracy']
metrics = ['accuracy']

#========================================================================================================
# Training the cnn:
#========================================================================================================

### loading training and validation data
x_train, y_train, x_val, y_val = load_data(training_path, validation_path)

### compiling the model
model = compile_model(model_id, run, my_models, GPUS, BOX_SIZE, loss, optimizer, metrics, output_path)

### training and validation
train_model(model, model_id, run, BATCH_SIZE, EPOCHS, ROTATIONS, BLUR, center_prob, x_train, y_train, x_val, y_val, BOX_SIZE, output_path)

### generating validation predictions
get_val_predictions(model, model_id, run, x_val, BATCH_SIZE, BLUR, center_prob, BOX_SIZE, output_path)

### results
get_plots(run, model_id, BLUR, loss, optimizer, learning_rate, training_path[3:-1], output_path, ROTATIONS)

print("Training completed!")





