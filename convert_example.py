import tensorflow as tf

import larq as lq
import larq_compute_engine as lce
import os

# Define a custom model



checkpoint_directory = os.path.join("./Keras_insightface","checkpoints","binarydensenet28_retina")
model_name = "keras_binarydensenet28_adaptativelr_retina_basic_agedb_30_epoch_16_binary_0.907167.h5"

basic_model = tf.keras.models.load_model(os.path.join(checkpoint_directory,model_name),compile=False)

lq.models.summary(basic_model)
# Note: Realistically, you would of course want to train your model before converting it!

# Convert our Keras model to a TFLite flatbuffer file
with open("new_binarydensenet28.tflite", "wb") as flatbuffer_file:
    flatbuffer_bytes = lce.convert_keras_model(basic_model)
    flatbuffer_file.write(flatbuffer_bytes)
