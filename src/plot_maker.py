from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import csv
from datetime import datetime
import time

# plot-maker

def timestamp():
  return str(datetime.now().time())

def get_history(model_id, output_path):
  """ parces the history CSV file """ 

  accuracy, loss, val_accuracy, val_loss = [], [], [], []

  with open(output_path + "/model_" + str(model_id) + "_history_log.csv") as hist_file:
    csv_reader = csv.DictReader(hist_file, delimiter=',')
    for row_values in csv_reader:
      accuracy.append(float(row_values['accuracy']))
      loss.append(float(row_values['loss']))
      val_accuracy.append(float(row_values['val_accuracy']))
      val_loss.append(float(row_values['val_loss']))
  
  return accuracy, loss, val_accuracy, val_loss

#graphing the accuracy and loss for both the training and test data
def get_plots(run, model_id, BLUR, loss, optimizer, learning_rate, data, output_path, rotations):
  """ creates simple plots of accuracy and loss for training and validation """

  parameter_text = "BLUR = " + str(BLUR) + "\n" + \
                   "loss = " + str(loss) + "\n" \
                   "optimizer = " + str(optimizer)[18:28] + ". \n" \
                   "learning rate = " + str(learning_rate) + "\n" \
                   "training data = " + str(data) + "\n" \
                   "rotations = " + str(rotations) 

  timestr = time.strftime("%m%d-%H%M%S")

  accuracy, loss, val_accuracy, val_loss = get_history(model_id, output_path)

  print("Making plots: ", timestamp(), "\n")

  # plotting accuracy 
  plt.plot(accuracy)
  plt.plot(val_accuracy)
  plt.title('Model ' +  model_id  +  ' Accuracy')
  plt.suptitle(str(datetime.now()), size = 7)
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['training', 'validation'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.28, 0.84), xycoords = 'axes fraction', size = 7) 
  plt.savefig(output_path + "/Accuracy_model_" + model_id + "_run_" + str(run) + "_" + timestr + ".pdf")
  plt.clf()

  # plotting loss
  plt.plot(loss)
  plt.plot(val_loss)
  plt.title('Model ' +  model_id  + ' Loss')
  plt.suptitle(str(datetime.now()), size = 7)
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training', 'validaton'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.28, 0.84), xycoords = 'axes fraction', size = 7)
  plt.savefig(output_path + "/loss_model_" + model_id + "_run_" + str(run) + "_" + timestr + ".pdf")

  #saving all data in a CSV file
  path = output_path + "/model_" + model_id + "_run_" + str(run) + "_" + timestr + ".csv"
  print("Starting to write CSV file:", timestamp())
  with open(path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Model_ID", "Epoch", "BLUR", "learning_rate", "rotations", "Acc_train", "Acc_val", "Loss_train", "Loss_val"])
    for i in range(0, len(accuracy)):
      writer.writerow([model_id, i+1, BLUR, learning_rate, rotations, accuracy[i], val_accuracy[i], loss[i], val_loss[i]])
  print("Finished writing CSV file:", timestamp())