import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


# 32m encoder model
model_32m = load_model('./encoder_model_32m.h5')
# model_32m.build(input_shape=(1, 14))
model_32m.export('./lite_model/saved_model_32m')
converter_32m = tf.lite.TFLiteConverter.from_saved_model('./lite_model/saved_model_32m')
tflite_model_32m = converter_32m.convert()
with open('model_32m.tflite', 'wb') as f:
    f.write(tflite_model_32m)

interpreter_32m = tf.lite.Interpreter(model_path='model_32m.tflite')
interpreter_32m.allocate_tensors()
tensor_details = interpreter_32m.get_tensor_details()


input_details = interpreter_32m.get_input_details()
output_details = interpreter_32m.get_output_details()

print("Input Details 32m:", input_details)
print("Output Details 32m:", output_details)
print(tensor_details)
