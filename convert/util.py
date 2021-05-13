from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

import tensorflow as tf

from config import FROZEN_PB_DIR, OUTPUT_DETECTION, OUTPUT_MASK


def freeze_session(sess, keep_var_names=None, output_names=None, clear_devices=True):
    graph = sess.graph

    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))

        output_names = output_names or []
        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def freeze_model(sess, model, name):
    frozen_graph = freeze_session(
        sess,
        output_names=[out.op.name for out in model.outputs][:4])
    directory = FROZEN_PB_DIR
    tf.train.write_graph(frozen_graph, directory, name , as_text=False)
    print("*"*80)
    print("Finish converting keras model to Frozen PB")
    print('PATH: ', FROZEN_PB_DIR)
    print("*" * 80)


def make_serving_ready(model_path, save_serve_path, version_number):
    import os
    import tensorflow as tf

    export_dir = os.path.join(save_serve_path, str(version_number))
    graph_pb = model_path

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    with tf.gfile.GFile(graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    sigs = {}

    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        input_image = g.get_tensor_by_name("input_image:0")
        input_image_meta = g.get_tensor_by_name("input_image_meta:0")
        input_anchors = g.get_tensor_by_name("input_anchors:0")

        output_detection = g.get_tensor_by_name(f"{OUTPUT_DETECTION}:0")
        output_mask = g.get_tensor_by_name(f"{OUTPUT_MASK}:0")

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input_image": input_image, 'input_image_meta': input_image_meta, 'input_anchors': input_anchors},
                {"mrcnn_detection/Reshape_1": output_detection, 'mrcnn_mask/Reshape_1': output_mask})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)

    builder.save()
    print("*" * 80)
    print("FINISH CONVERTING FROZEN PB TO SERVING READY")
    print("PATH:", export_dir)
    print("*" * 80)