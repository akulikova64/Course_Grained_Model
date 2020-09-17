from generators import predict_dataGenerator
import numpy as np

import time
from datetime import datetime
def timestamp():
  return str(datetime.now().time())

# makes predictions

def predict(model, pre_boxes, batch_size, BLUR, center_prob, box_size):
  """ for making predictions on any dataset """
  predictions = model.predict_generator(
        generator = predict_dataGenerator(pre_boxes, batch_size, BLUR, center_prob, box_size),
        steps = len(pre_boxes)/batch_size,
        verbose = 1)

  return predictions

def get_val_predictions(model, model_id, run, x_val, BATCH_SIZE, BLUR, center_prob, BOX_SIZE):
  """ generating validation predicitons """

  print("Starting to predict:", timestamp(), "\n")
  predictions = predict(model, x_val, BATCH_SIZE, BLUR, center_prob, BOX_SIZE)
  timestr = time.strftime("%m%d-%H%M%S")
  np.save("../output/predictions_model_" + model_id + "_run_" + str(run) + "_" + timestr + ".npy", predictions)
  print("Finished predicting:", timestamp(), "\n")


