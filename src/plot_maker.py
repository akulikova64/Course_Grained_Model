from matplotlib import pyplot as plt
from datetime import datetime

def timestamp_2():
  date_str = ""
  date = str(datetime).split(" ")
  for item in date:
    date_str += ("_" + item + "_")

  return date_str

# plot-maker

#graphing the accuracy and loss for both the training and test data
def get_plots(history, model_id, BLUR, loss, optimizer, learning_rate, data):
  """ creates simple plots of accuracy and loss for training and validation """

  parameter_text = "BLUR = " + str(BLUR) + "\n" + \
                   "loss = " + str(loss) + "\n" \
                   "optimizer = " + str(optimizer)[18:28] + ". \n" \
                   "learning rate = " + str(learning_rate) + "\n" \
                   "training data = " + str(data)

  #summarize history for accuracy 
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model ' +  model_id  +  ' Accuracy')
  plt.suptitle(str(datetime.now()))
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['training', 'validation'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.30, 0.78), xycoords = 'axes fraction')
  plt.savefig("../output/Accuracy_model_" + model_id + timestamp_2() + ".pdf")
  plt.clf()

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model ' +  model_id  + ' Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training', 'validaton'], loc = 'upper left')
  plt.annotate(parameter_text, xy = (0.30, 0.78), xycoords = 'axes fraction')
  plt.savefig("../output/loss_model_" + model_id + timestamp_2() + ".pdf")

  