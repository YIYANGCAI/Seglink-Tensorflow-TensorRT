import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

input_image = np.random.randint(0,255,(1,384,384,3))

with tf.Session() as sess:
    saver = tf.train.import_meta_graph('ckpt/model.ckpt-5882.meta',clear_devices=True)
    saver.restore(sess,'ckpt/model.ckpt-5882')
    graph = sess.graph

    sess.run(tf.global_variables_initializer())

    input_name = graph.get_tensor_by_name('ones:0')

    concat_1 = graph.get_tensor_by_name('seglink_layers/concat_1:0')
    softmax = graph.get_tensor_by_name('seglink_layers/softmax/Reshape_1:0')
    softmax_1 = graph.get_tensor_by_name('seglink_layers/softmax_1/Reshape_1:0')

    ckpt_res = sess.run([concat_1,softmax,softmax_1],feed_dict={input_name:input_image})

    print(ckpt_res)
    


with tf.Session() as sess_pb:
    with gfile.FastGFile('new.pb', 'rb') as f: 
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess_pb.graph.as_default()
        tf.import_graph_def(graph_def, name='')  

    sess_pb.run(tf.global_variables_initializer())

    graph = sess_pb.graph 


    input_shape_name = graph.get_tensor_by_name('ones/shape_as_tensor:0')
    input_data_name  = graph.get_tensor_by_name('ones/Const:0')
    input_name = graph.get_tensor_by_name('ones:0')

    concat_1 = graph.get_tensor_by_name('seglink_layers/concat_1:0')
    softmax = graph.get_tensor_by_name('seglink_layers/softmax/Reshape_1:0')
    softmax_1 = graph.get_tensor_by_name('seglink_layers/softmax_1/Reshape_1:0')

    pb_res = sess_pb.run([concat_1,softmax,softmax_1],feed_dict={input_name:input_image})

    print(ckpt_res)

