import tensorflow as tf

import numpy as np

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('ckpt/model.ckpt-5882.meta',clear_devices=True)
    saver.restore(sess,'ckpt/model.ckpt-5882')
    graph = tf.get_default_graph() # 获得默认的图

    input_graph_def = graph.as_graph_def()  
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,
        output_node_names=['seglink_layers/softmax/Reshape_1','seglink_layers/softmax_1/Reshape_1','seglink_layers/concat_1'])

    with tf.gfile.FastGFile('new.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
