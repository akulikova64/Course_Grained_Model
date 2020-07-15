from matplotlib import pyplot as plt
# plot-maker

#graphing the accuracy and loss for both the training and test data
def get_plots(history, model_id):
  """ creates simple plots of accuracy and loss for training and validation """

  print("Making plots: " + datetime.now().time())

  #summarize history for accuracy 
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['training', 'validation'], loc = 'upper left')
  plt.savefig("Accuracy_model_" + model_id + ".pdf")
  plt.clf()

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['training', 'validaton'], loc = 'upper left')
  plt.savefig("loss_model_" + model_id + ".pdf")