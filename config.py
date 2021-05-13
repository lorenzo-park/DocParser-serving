from docparser.utils.data_utils import DocsDataset
from docparser.utils.experiment_utils import DocparserDefaultConfig

import os


# keras model file path
WEIGHT_PATH = '/home/lorenzo-lab/project/DocParser-serving/model/docparser.h5'
MODEL_DIR = os.path.dirname(WEIGHT_PATH)

# Path where the Frozen PB will be save
FROZEN_PB_DIR = '/home/lorenzo-lab/project/DocParser-serving/frozen_model/'

# PATH where to save serving model
SERVING_MODEL_PATH = '/home/lorenzo-lab/project/DocParser-serving/serving_model'

# Name for the Frozen PB name
FROZEN_NAME = 'mask_frozen_graph.pb'


# Version of the serving model
VERSION_NUMBER = 1

SIGNATURE_NAME = "serving_default"
INPUT_IMAGE = "input_image"
INPUT_IMAGE_META = "input_image_meta"
INPUT_ANCHORS = "input_anchors"
OUTPUT_DETECTION = 'mrcnn_detection/Reshape_1'
OUTPUT_MASK = 'mrcnn_mask/Reshape_1'

class InferenceConfig(DocparserDefaultConfig):
    NAME = 'docparser_inference'
    DETECTION_MAX_INSTANCES = 200
    IMAGE_RESIZE_MODE = "square"
    NUM_CLASSES = len(DocsDataset.ALL_CLASSES) + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
