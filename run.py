from bentoml.frameworks.tensorflow import TensorflowSavedModelArtifact
from bentoml.adapters import FileInput
from PIL import Image

import bentoml
import io
import tensorflow as tf
import numpy as np

from config import InferenceConfig, INPUT_IMAGE, INPUT_IMAGE_META, INPUT_ANCHORS
from docparser import stage2_structure_parser
from docparser.utils.data_utils import DocsDataset
from docparser.utils.eval_utils import convert_bbox_list_to_save_format
from util import ForwardModel

tf.compat.v1.enable_eager_execution() # required


@bentoml.env(pip_packages=['tensorflow'])
@bentoml.artifacts([TensorflowSavedModelArtifact('trackable')])
class DocparserService(bentoml.BentoService):
    def __init__(self):
        super().__init__()
        model_config = InferenceConfig()
        self.preprocess_obj = ForwardModel(model_config)

    def preprocess(self, image):
        images = np.expand_dims(image, axis=0)[:,:,:,:3]
        molded_images, image_metas, windows = self.preprocess_obj.mold_inputs(images)
        molded_images = molded_images.astype(np.float32)
        image_shape = molded_images[0].shape

        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        anchors = self.preprocess_obj.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (images.shape[0],) + anchors.shape)

        self.images = images
        self.molded_images = molded_images
        self.windows = windows

        # response body format row wise.
        return {
            INPUT_IMAGE: molded_images[0].tolist(),
            INPUT_IMAGE_META: image_metas[0].tolist(),
            INPUT_ANCHORS: anchors[0].tolist()
        }


    @bentoml.api(input=FileInput(), batch=True)
    def predict(self, inputs):
        # print(inputs)
        inputs = [self.preprocess(Image.open(io.BytesIO(i.read()))) for i in inputs]
        inputs = {
            INPUT_ANCHORS: [i[INPUT_ANCHORS] for i in inputs],
            INPUT_IMAGE: [i[INPUT_IMAGE] for i in inputs],
            INPUT_IMAGE_META: [i[INPUT_IMAGE_META] for i in inputs],
        }
        loaded_func = self.artifacts.trackable.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        input_anchors = tf.constant(inputs[INPUT_ANCHORS], dtype=tf.float32)
        input_image = tf.constant(inputs[INPUT_IMAGE], dtype=tf.float32)
        input_image_meta = tf.constant(inputs[INPUT_IMAGE_META], dtype=tf.float32)

        pred = loaded_func(
            input_anchors=input_anchors,
            input_image=input_image,
            input_image_meta=input_image_meta,
        )
        results = self.preprocess_obj.result_to_dict(self.images, self.molded_images, self.windows, pred)
        outputs = []
        structure_parser = stage2_structure_parser.StructureParser()
        for idx, r in enumerate(results):
            pred_bboxes = r['rois']
            pred_class_ids = r['class_ids']
            pred_scores = r['scores']
            classes_with_background = ['background'] + DocsDataset.ALL_CLASSES

            orig_shape = list(self.images[idx].shape)

            num_preds = len(pred_bboxes)
            prediction_list = []
            for pred_nr in range(num_preds):
                class_name = classes_with_background[pred_class_ids[pred_nr]]
                pred_bbox = pred_bboxes[pred_nr]
                pred_score = pred_scores[pred_nr]
                prediction_list.append({'pred_nr': pred_nr, 'class_name': class_name, 'pred_score': pred_score,
                                        'bbox_orig_coords': pred_bbox, 'orig_img_shape': orig_shape})

            img_relations_dict = structure_parser.create_structure_for_doc_from_list({'prediction_list': prediction_list, 'orig_img_shape': orig_shape}, do_postprocessing=True)
            predictions_dict = convert_bbox_list_to_save_format(img_relations_dict['all_bboxes'])
            try:
                del img_relations_dict['all_bboxes']
            except KeyError:
                pass
            predictions_dict["relations"] = img_relations_dict
            outputs.append(predictions_dict)
        return outputs

# bento_svc = DocparserService()
# bento_svc.pack("trackable", "serving_model/1")
# saved_path = bento_svc.save()