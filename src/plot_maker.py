from matplotlib import pyplot as plt
import csv
from datetime import datetime
import time

def timestamp():
  return str(datetime.now().time())

# plot-maker

def get_history(model_id):
  # upload history object here. 
  return history

#graphing the accuracy and loss for both the training and test data
def get_plots(run, model_id, BLUR, loss, optimizer, learning_rate, data):
  """ creates simple plots of accuracy and loss for training and validation """

  parameter_text = "BLUR = " + str(BLUR) + "\n" + \
                   "loss = " + str(loss) + "\n" \
                   "optimizer = " + str(optimizer)[18:28] + ". \n" \
                   "learning rate = " + str(learning_rate) + "\n" \
                   "training data = " + str(data)

  timestr = time.strftime("%m%d-%H%M%S")
  history = get_history(model_id)

  #summarize history for accuracy 
  print("Making plots: ", timestamp(), "\n")

  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model ' +  model_id  +  ' Accuracy')
  plt.suptitle(str(datetime.now()), size = 7)
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['training', 'validation'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.28, 0.84), xycoords = 'axes fraction', size = 7)
  plt.savefig("../output/Accuracy_model_" + model_id + "_run_" + str(run) + "_" + timestr + ".pdf")
  plt.clf()

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model ' +  model_id  + ' Loss')
  plt.suptitle(str(datetime.now()), size = 7)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training', 'validaton'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.28, 0.84), xycoords = 'axes fraction', size = 7)
  plt.savefig("../output/loss_model_" + model_id + "_run_" + str(run) + "_" + timestr + ".pdf")

  #saving data in a CSV file
  path = "../output/model_" + model_id + "_run_" + str(run) + "_" + timestr + ".csv"
  print("Starting to write CSV file:", timestamp())
  with open(path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model_ID", "Epoch", "BLUR", "learning_rate", "Acc_train", "Acc_val", "Loss_train", "Loss_val"])
    for i in range(0, len(history.history['accuracy'])):
      writer.writerow([model_id, i+1, BLUR, learning_rate, history.history['accuracy'][i], history.history['val_accuracy'][i], history.history['loss'][i], history.history['val_loss'][i]])
  print("Finished writing CSV file:", timestamp())

  print(model.summary(), "\n")
  