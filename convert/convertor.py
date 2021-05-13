"""
CREDIT: https://github.com/bendangnuksung/mrcnn_serving_ready/blob/de9cd824e6e3a108dcd6af50a4a377afc3f24d08/main.py
"""
import sys
sys.path.append('.')

import os

import mrcnn.model as modellib
import tensorflow as tf
import keras.backend as K

from convert.util import freeze_model, make_serving_ready
from config import InferenceConfig, FROZEN_PB_DIR, FROZEN_NAME, SERVING_MODEL_PATH, VERSION_NUMBER, WEIGHT_PATH


sess = tf.Session()
K.set_session(sess)

config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir=WEIGHT_PATH)
model.load_weights(WEIGHT_PATH, by_name=True)

# Converting keras model to PB frozen graph
freeze_model(sess, model.keras_model, FROZEN_NAME)

# Now convert frozen graph to Tensorflow Serving Ready
make_serving_ready(os.path.join(FROZEN_PB_DIR, FROZEN_NAME),
                     SERVING_MODEL_PATH,
                     VERSION_NUMBER)

print("COMPLETED")