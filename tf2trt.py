import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

with tf.Session() as sess:
    # First deserialize your frozen graph:
    with tf.gfile.GFile("./new.pb", 'rb') as f:
        frozen_graph = tf.GraphDef()
        frozen_graph.ParseFromString(f.read())
    # Now you can create a TensorRT inference graph from your
    # frozen graph:
    converter = trt.TrtGraphConverter(
	    input_graph_def=frozen_graph,
	    nodes_blacklist=['seglink_layers/softmax/Reshape_1','seglink_layers/softmax_1/Reshape_1','seglink_layers/concat_1']) #output nodes
    trt_graph = converter.convert()
    # Import the TensorRT graph into a new graph and run:
    output_node = tf.import_graph_def(
        trt_graph,
        return_elements=['seglink_layers/softmax/Reshape_1','seglink_layers/softmax_1/Reshape_1','seglink_layers/concat_1'])
    sess.run(output_node)
