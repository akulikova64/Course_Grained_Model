import numpy as np
import collections
from generators import train_dataGenerator
from generators import test_val_dataGenerator

try:
  import keras
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Activation, Flatten
  from keras.layers import Convolution3D
  from keras.optimizers import Adam
  from keras.callbacks import Callback
  from keras.models import load_model
  from keras.utils import multi_gpu_model

except ImportError:
  import tensorflow
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import Convolution3D
  from tensorflow.keras.callbacks import Callback
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models import load_model
  from tensorflow.keras.utils import multi_gpu_model

# Training and testing
# global dict for decoding amino acid centers:
AA_DICT = {0:'H', 1:'E', 2:'D', 3:'R', 4:'K', 5:'S', 6:'T', 7:'N', 8:'Q', 9:'A', 10:'V', 11:'L', 12:'I', 13:'M', 14:'F', 15:'Y', 16:'W', 17:'P', 18:'G', 19:'C'}
  
def check_for_missing_classes(centers_dict, aa_dict):
  """ finds and reports any missing aa from a dataset"""
  
  missing_aa = []
  if len(centers_dict) < 20:
    for i in range(0, 20):
      if i not in centers_dict:
        missing_aa.append(aa_dict[i])

  return missing_aa

def normalize_classes(centers):
  """ calculates class weights for normalizing the classes"""

  # 1) counting the number of occurrences of each amino acid in the training set (key is aa, value is count)
  centers_dict = collections.Counter(centers)
  
  # 2) checking for missing amino acid classes and printing a warning
  missing_aa = check_for_missing_classes(centers_dict, AA_DICT)
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

# training the model
def train_model(model, callbacks_list, batch_size, epochs, rotations, BLUR, center_prob, x_train, y_train, x_val, y_val, box_size):
  """ calling the model to train """

  class_weight = normalize_classes(y_train)

  history = model.fit_generator(
            generator = train_dataGenerator(x_train, y_train, batch_size, rotations, BLUR, center_prob, box_size),
            validation_data = test_val_dataGenerator(x_val, y_val, batch_size, BLUR, center_prob, box_size),
            validation_steps = 20,
            steps_per_epoch = len(x_train)/batch_size, 
            epochs = epochs, 
            verbose = 1,
            class_weight = class_weight,
            callbacks = callbacks_list
          )

  return history

# returns testing results
def get_testing_results(model, batch_size, x_test, y_test):
  """ testing the trained model """

  score = model.evaluate(x_test, y_test, verbose = 0, steps = int(len(x_test)/batch_size))  
  #score = model.evaluate_generator(x_test, y_test, verbose = 1, steps = int(len(x_test)/batch_size))

  return score
