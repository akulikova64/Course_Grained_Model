import numpy as np
import collections
import math
from generators import dataGenerator_rot
from generators import dataGenerator_no_rot
from box_maker import make_one_box

try:
  import keras
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Activation, Flatten
  from keras.layers import Convolution3D
  from keras.optimizers import Adam
  from keras.callbacks import Callback
  from keras.models import load_model
  from keras.utils import multi_gpu_model
  from keras.utils import to_categorical
  from keras.callbacks import ModelCheckpoint
  from keras.callbacks import CSVLogger

except ImportError:
  import tensorflow
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import Convolution3D
  from tensorflow.keras.callbacks import Callback
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models import load_model
  from tensorflow.keras.utils import multi_gpu_model
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import ModelCheckpoint
  from tensorflow.keras.callbacks import CSVLogger

from datetime import datetime
def timestamp():
  return str(datetime.now().time())

# Training and testing
# global dict for decoding amino acid centers:
AA_DICT = {0:'H', 1:'E', 2:'D', 3:'R', 4:'K', 5:'S', 6:'T', 7:'N', 8:'Q', 9:'A', 10:'V', 11:'L', 12:'I', 13:'M', 14:'F', 15:'Y', 16:'W', 17:'P', 18:'G', 19:'C'}
  
def check_for_missing_classes(centers_dict):
  """ finds and reports any missing aa from a dataset"""
  
  missing_aa = []
  if len(centers_dict) < 20:
    for i in range(0, 20):
      if i not in centers_dict:
        missing_aa.append(AA_DICT[i])

  return missing_aa

def normalize_classes(centers):
  """ calculates class weights for normalizing the classes"""

  # 1) counting the number of occurrences of each amino acid in the training set (key is aa, value is count)
  centers_dict = collections.Counter(centers)
  
  # 2) checking for missing amino acid classes and printing a warning
  missing_aa = check_for_missing_classes(centers_dict)
  for aa in missing_aa:
    print(aa, "is missing from dataset!")  
  print(str(len(missing_aa)), "aa type(s) missing from dataset!")

  # 3) calculating the weights and saving to dictionary (key is aa, value is class_weight)
  total = len(centers)
  num_classes = len(centers_dict)
  class_weights = {}

  for aa in centers_dict:
    # the number of aa that should be in each class is divided by how many actually are in each class
    class_weights[aa] = (1/centers_dict[aa])*(total)/num_classes

  print("class weights:")
  print(class_weights)
    
  return class_weights

def save_checkpoint(model_id, run, output_path):
  """ saves checkpoint in case of interruption """

  checkpoint_path = output_path + "/model_ckeckpoint" + model_id + "_run_" + str(run) + ".h5"
  checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', mode='max', save_best_only=False)
  print("Checkpoint file created:", timestamp(), "\n")

  return checkpoint

def save_csv_logger(model_id, output_path):
  """ saves epoch history as CSV file """

  csv_logger_path = output_path + "/model_" + model_id + "_history_log.csv"
  csv_logger = CSVLogger(csv_logger_path, append=True)
  print("History CSV file loaded and ready, starting to train:", timestamp(), "\n")

  return csv_logger

def save_model(model, model_id, run, output_path):
  """ saves model with weights """

  current_model = output_path + "/model_" + model_id + "_run_" + str(run) + ".h5"
  model.save(current_model)
  print("Saved current model:", timestamp(), "\n")

# training and saving the model
def train_model(model, model_id, run, batch_size, epochs, rotations, BLUR, center_prob, x_train, y_train, x_val, y_val, box_size, output_path):
  """ calling the model to train """

  class_weight = normalize_classes(y_train)
  checkpoint = save_checkpoint(model_id, run, output_path)
  csv_logger = save_csv_logger(model_id, output_path)

  print("Starting to train:", timestamp(), "\n")
  
  if rotations > 0:
    #steps_per_epoch = math.ceil(len(x_train)/(batch_size/rotations))
    #validation_steps = math.ceil(len(x_val)/batch_size)
    a = 4
    steps_per_epoch = math.ceil(a/rotations)
    validation_steps = a
    model.fit_generator(
        generator = dataGenerator_rot(x_train, y_train, batch_size, rotations, BLUR, center_prob, box_size),
        validation_data = dataGenerator_no_rot(x_val, y_val, batch_size, BLUR, center_prob, box_size),
        validation_steps = validation_steps, 
        steps_per_epoch = steps_per_epoch, 
        max_queue_size = 0,
        epochs = epochs, 
        verbose = 1,
        class_weight = class_weight,
        callbacks = [checkpoint, csv_logger])
  else:
    steps_per_epoch = math.ceil(len(x_train)/batch_size)
    validation_steps = math.ceil(len(x_val)/batch_size)
    model.fit_generator(
        generator = dataGenerator_no_rot(x_train, y_train, batch_size, BLUR, center_prob, box_size),
        validation_data = dataGenerator_no_rot(x_val, y_val, batch_size, BLUR, center_prob, box_size),
        validation_steps = validation_steps, 
        steps_per_epoch = steps_per_epoch, 
        epochs = epochs, 
        verbose = 1,
        class_weight = class_weight,
        callbacks = [checkpoint, csv_logger])

  print("Finished training and validation:", timestamp(), "\n")
  
  save_model(model, model_id, run, output_path)
  print("Saved model:", timestamp(), "\n")
  print(model.summary(), "\n")


