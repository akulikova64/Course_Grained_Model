# course grained cnn
from train_test import train_model
from train_test import get_testing_results
from train_test import load_model
from plot_maker import get_plots
from predictor import predict
import models

try:
  from keras.models import load_model
  from keras.optimizers import Adam
  from keras.callbacks import ModelCheckpoint
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np
import time
import sys
from datetime import datetime
def timestamp():
  return str(datetime.now().time())


#========================================================================================================
# Setting the variables, parameters and data paths/locations:
#========================================================================================================

### variables
EPOCHS = 1 # iterations through the data
BOX_SIZE = 9
ROTATIONS = 4 # number of box rotations per box
BATCH_SIZE = 20 # batch_size must be divisible by "ROTATIONS"
GPUS = 1 # max is 4 GPUs
BLUR = False
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel
model_id = "30"
learning_rate = 0.0001

### data paths/locations
training_path = "../boxes/"
validation_path = "../validation/"
testing_path = "../testing/"

### best models:
my_models = {"27": models.model_27, "28": models.model_28, "29": models.model_29, "30": models.model_30, "12": models.model_12, "13": models.model_13, "14": models.model_14, "15": models.model_15, "20": models.model_20, "21": models.model_21, "22": models.model_22, "23": models.model_23, "24": models.model_24, "25": models.model_25}

### setting parameters for training
loss ='categorical_crossentropy'
optimizer = Adam(lr = learning_rate)
metrics = ['accuracy']

#========================================================================================================
# Training, testing and saving the cnn:
#========================================================================================================

### loading traianing and validation data
print("\nStarting to load training data:", timestamp())
x_train = np.load(training_path + "boxes_train.npy", allow_pickle = True).tolist()
y_train = np.load(training_path + "centers_train.npy", allow_pickle = True).tolist()
print("Finished loading training data:", timestamp())
x_val = np.load(validation_path + "boxes_normalized.npy", allow_pickle = True).tolist()
y_val = np.load(validation_path + "centers_normalized.npy", allow_pickle = True).tolist()
print("Finished loading validation data:", timestamp())

### compiling the model 
model = my_models[model_id](GPUS, BOX_SIZE)
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
print("Model compiled:", timestamp(), "\n")

### saving checkpoint
model_path = "../output/model_" + model_id + ".h5"
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', mode='max', save_best_only=False)
callbacks_list = [checkpoint]
print("Checkpoint file created, starting to train:", timestamp(), "\n")

### training and validation
history = train_model(model, model_path, callbacks_list, BATCH_SIZE, EPOCHS, ROTATIONS, BLUR, center_prob, x_train, y_train, x_val, y_val, BOX_SIZE)
print("Finished training and validation:", timestamp(), "\n")

### saving current model
timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = "../output/model_" + model_id + "_" + timestr + ".h5"
model.save(model_name)
print("Saved current model:", timestamp(), "\n")

### saving validation predictions
print("Starting to predict:", timestamp(), "\n")
predictions = predict(model, x_val, BATCH_SIZE, BLUR, center_prob, BOX_SIZE)
np.save("../output/predictions_model_" + str(model_id) + ".npy", predictions)
print("Finished predicting:", timestamp(), "\n")
 
### testing
print("Finished training, loading test data:", timestamp())
x_test = np.load(testing_path + "boxes_normalized.npy", allow_pickle = True).tolist()
y_test = np.load(testing_path + "centers_normalized.npy", allow_pickle = True).tolist()
print("Finished loading test data, testing:", timestamp())
score = get_testing_results(model, BOX_SIZE, BATCH_SIZE, BLUR, center_prob, x_test, y_test)
print("Finished testing:", timestamp(), "\n")

### results
get_plots(history, model_id, BLUR, loss, optimizer, learning_rate, training_path[3:-1])
print("Making plots: ", timestamp(), "\n")
print(model.summary(), "\n")
print('Test loss:', score[0])
print('Test accuracy:', score[1])









