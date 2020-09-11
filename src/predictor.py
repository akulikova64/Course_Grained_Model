from generators import predict_dataGenerator

# makes predictions

def predict(model, pre_boxes, batch_size, BLUR, center_prob, box_size):
  predictions = model.predict_generator(
        generator = predict_dataGenerator(pre_boxes, batch_size, BLUR, center_prob, box_size),
        steps = len(pre_boxes)/batch_size,
        verbose = 1)

  return predictions


