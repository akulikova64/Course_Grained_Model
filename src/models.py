try:
  import keras
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Activation, Flatten
  from keras.layers import BatchNormalization
  from keras.layers import Convolution3D
  from keras.optimizers import Adam
  from keras.callbacks import Callback
  from keras.models import load_model
  from keras.utils import multi_gpu_model

except ImportError:
  import tensorflow
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import BatchNormalization
  from tensorflow.keras.layers import Convolution3D
  from tensorflow.keras.callbacks import Callback
  from tensorflow.keras.optimizers import Adam
  from tensorflow.keras.models import load_model
  from tensorflow.keras.utils import multi_gpu_model

# models

# cnn model structure
def model_1(GPUS = 1, box_size = 9):
  """ model with three conv layers and one dense layer """
  model = Sequential()
  model.add(Convolution3D(32, kernel_size = (3, 3, 3), strides = (1, 1, 1), activation = 'relu', input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(Convolution3D(32, (3, 3, 3), activation = 'relu'))
  model.add(Convolution3D(32, (3, 3, 3), activation = 'relu'))
  model.add(Convolution3D(32, (3, 3, 3), activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(500, activation = 'relu')) # 500 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

def model_2(GPUS = 1):
  """ model with 5 conv layers and one large dense layer """
  model = Sequential()
  model.add(Convolution3D(500, kernel_size = (3, 3, 3), strides = (1, 1, 1), activation = 'relu', input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(Convolution3D(100, (3, 3, 3), activation = 'relu'))
  model.add(Convolution3D(50, (3, 3, 3), activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 500 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_3(GPUS = 1):
  """ model with three medium conv layers, one dense layer and more neurons """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), activation = 'relu', input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(Convolution3D(60, (3, 3, 3), activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3), activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 500 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_4(GPUS = 1):
  """ model with three medium conv layers, one dense layer and more neurons """
  model = Sequential()
  model.add(Convolution3D(30, kernel_size = (1, 1, 1), strides = (1, 1, 1), activation = 'relu', input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(Convolution3D(60, (3, 3, 3), activation = 'relu')) 
  model.add(Convolution3D(60, (3, 3, 3), activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3), activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 500 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_5(GPUS = 1):
  """ Added Batch Normalization, 3 dense layers """
  model = Sequential()
  model.add(Convolution3D(35, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3)))
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_6(GPUS = 1):
  """ No batch norm, three dense layers """
  model = Sequential()
  model.add(Convolution3D(35, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3))) 
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3)))
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_7(GPUS = 1):
  """ batch norm and one dense layer """
  model = Sequential()
  model.add(Convolution3D(35, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(35, (3, 3, 3)))
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_8(GPUS = 1):
  """ One convolutional layer """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_9(GPUS = 1):
  """ two convolutional layers """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_10(GPUS = 1):
  """ three convolutional layers """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_11(GPUS = 1):
  """ four convolutional layers """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(300, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(100, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_12(GPUS = 1):
  """ two dense layers: 1000, 800 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(800, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_13(GPUS = 1):
  """ three dense layers: 1000, 800, 600 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(800, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(600, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_14(GPUS = 1):
  """ four dense layers: 1000, 800, 600, 400 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(800, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(600, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(400, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_15(GPUS = 1):
  """ five dense layers: 1000, 800, 600, 400, 200 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(800, activation = 'relu')) # 200 nodes in the last hidden layer
  model.add(Dense(600, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(400, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(200, activation = 'relu')) # 100 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)

  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_16(GPUS = 1):
  """one dense: 1500 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1500, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_17(GPUS = 1):
  """ one dense: 2000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(2000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_18(GPUS = 1):
  """ one dense: 2500 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(2500, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_19(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000, activation = 'relu')) # 300 nodes in the last hidden layer
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_20(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_21(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (3, 3, 3))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dropout(0.5))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_22(GPUS = 1, box_size = 9):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_23(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(2000, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(4000, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(6000, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_24(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(100, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(150, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_25(GPUS = 1):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (9, 9, 9, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(100, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(150, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(300, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(3000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_26(GPUS = 1, box_size = 9):
  """ one dense: 3000 """
  model = Sequential()
  model.add(Convolution3D(50, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(100, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(150, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(200, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_27(GPUS = 1, box_size = 9):
  """ one dense: 20  """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(20)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model
  
def model_28(GPUS = 1, box_size = 9):
  """ one dense: 50 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(50)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_29(GPUS = 1, box_size = 9):
  """ one dense: 100 nodes"""
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(100)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_30(GPUS = 1, box_size = 9):
  """ one dense: 1000 """
  model = Sequential()
  model.add(Convolution3D(60, kernel_size = (3, 3, 3), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Convolution3D(60, (2, 2, 2))) 
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_31(GPUS = 1, box_size = 9):
  """ model with minimal convolutions """
  model = Sequential()
  model.add(Convolution3D(500, kernel_size = (2, 2, 2), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(1000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_32(GPUS = 1, box_size = 3):
  """ model for a small 3x3x3 box, no conv """
  model = Sequential()
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_33(GPUS = 1, box_size = 9):
  """ model with no convolutions """
  model = Sequential()
  model.add(Flatten()) 
  model.add(Dense(1000)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

def model_34(GPUS = 1, box_size = 3):
  """ model for a small 3x3x3 box with conv """
  model = Sequential()
  model.add(Convolution3D(500, kernel_size = (2, 2, 2), strides = (1, 1, 1), input_shape = (box_size, box_size, box_size, 20))) # 32 output nodes, kernel_size is your moving window, activation function, input shape = auto calculated
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Flatten()) # now our layers have been combined to one
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(500)) # 300 nodes in the last hidden layer
  model.add(BatchNormalization())
  model.add(Activation(activation = 'relu'))
  model.add(Dense(20, activation = 'softmax')) # output layer has 20 possible classes (amino acids 0 - 19)
  
  if GPUS >= 2:
    model = multi_gpu_model(model, gpus=GPUS)

  return model

