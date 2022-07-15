import os

import tensorflow as tf
from larq.layers import *
import larq_compute_engine as lce

checkpoint_directory = os.path.join(".","checkpoints","binaryfacenet_best2")
model_name = "keras_binaryfacenet_retina_basic_agedb_30_epoch_19_binary_0.751167.h5"
convert_name="test.tflite"

basic_model = tf.keras.models.load_model(os.path.join(checkpoint_directory,model_name),compile=False)
tflite_model = lce.convert_keras_model(basic_model)


with open(convert_name,"wb") as f:
    f.write(tflite_model)

