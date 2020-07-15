# course grained cnn with rotations
from box_maker import get_box_list
from box_maker import get_test_data
import models
from train_test import train_model
from train_test import get_testing_results
from train_test import load_model
from plot_maker import get_plots
from datetime import datetime

try:
  from keras.models import load_model
  from keras.optimizers import Adam
except ImportError:
  from tensorflow.keras.models import load_model
  from tensorflow.keras.optimizers import Adam

#========================================================================================================
# Setting the variables, parameters and file names:
#========================================================================================================

### variables
EPOCHS = 3 # iterations through the data
ROTATIONS = 4 # number of box rotations per box
BATCH_SIZE = 20 # batch_size must be divisible by "ROTATIONS"
GPUS = 4 # max is 4 GPUs
BLUR = True
center_prob = 0.44 if BLUR else 1 # probability of amino acid in center voxel
model_id = "5"
learning_rate = 0.1

### data paths/locations
training_path = "../boxes_tenth/"
validation_path = "../boxes_38/"
testing_path_x = "../testing/boxes_test.npy"
testing_path_y = "../testing/centers_test.npy"

### models:
my_models = {"3": model_3(), "5": model_5()}

### setting parameters for training
loss ='categorical_crossentropy'
optimizer = Adam(lr = learning_rate)
metrics = ['accuracy']

#========================================================================================================
# Training, testing and saving the cnn:
#========================================================================================================

### training and validation
x_train, y_train = get_box_list(training_path) # preparing training data (boxes, centers)
print("Finished loading training data: " + datetime.now().time())
x_val, y_val = get_box_list(validation_path) # preparing validation data (boxes, centers)
print("Finished loading validation data: " + datetime.now().time())
model = models.my_models[model_id](GPUS)

print("Model compiled, starting to train: " + datetime.now().time())
history = train_model(model, BATCH_SIZE, EPOCHS, ROTATIONS, BLUR, center_prob, x_train, y_train, x_val, y_val)

### testing
print("Finished training, loading test data: " + datetime.now().time())
x_test, y_test = get_test_data(testing_path_x, testing_path_y)
print("Finished loading test data, testing: " + datetime.now().time())
score = get_testing_results(model, BATCH_SIZE, x_test, y_test)
print("Finished testing: " + datetime.now().time())

### saving and loading trained model
model_name = "../output/model_" + model_id + ".h5"
model.save(model_name)
model = load_model(model_name)
  
### results
get_plots(history, model_id)
print()
print(model.summary())
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])









