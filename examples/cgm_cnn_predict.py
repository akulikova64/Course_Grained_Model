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
from datetime import datetime
def timestamp():
  return str(datetime.now().time())

# this code is for runnning cnn in prediction mode


model_id = "30"

### saving and loading trained model
timestr = time.strftime("%Y%m%d-%H%M%S")
model_name = "../output/model_" + model_id + "_" + timestr + ".h5"
model.save(model_name)
model = load_model(model_name)
print("Loaded model:", timestamp(), "\n")