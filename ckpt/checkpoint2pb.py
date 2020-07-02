import tensorflow as tf
from tensorflow import graph_util
from tensorflow.python import pywrap_tensorflow
import os


import tensorflow as tf
from tensorflow.summary import FileWriter

def create_graph(out_pb_path):
    # 读取并创建一个图graph来存放训练好的模型
    with tf.gfile.FastGFile(out_pb_path, 'rb') as f:
        # 使用tf.GraphDef() 定义一个空的Graph
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # Imports the graph from graph_def into the current default Graph.
        tf.import_graph_def(graph_def, name='')
 
def check_pb_out_name(out_pb_path, result_file):
    create_graph(out_pb_path)
    tensor_name_list = [tensor.name for tensor in
                        tf.get_default_graph().as_graph_def().node]
    with open(result_file, 'w+') as f:
        for tensor_name in tensor_name_list:
            f.write(tensor_name+'\n')

def checkNode_2(checkpoint_path):
    sess = tf.Session()
    tf.train.import_meta_graph(checkpoint_path+'.meta')
    FileWriter("__tb", sess.graph)
    print("Success!")

def checkNode_1(checkpoint_path):
    reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map=reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        print('tensor_name: {}'.format(key))

def getOutNodes(file):
    nodes = []
    with open(file, "r") as load_f:
        for tensor in load_f.readlines():
            tensor = tensor.strip('\n')
            nodes.append(tensor)
            print("\t{}".format(tensor))
    return nodes

def freezeGraph(input_checkpoint, output_nodes_names, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    output_node_names = output_nodes_names
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint) #恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names)# 如果有多个输出节点，以逗号隔开
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点
 
        # for op in graph.get_operations():
        #     print(op.name, op.values())

if __name__ == '__main__':
    nodes = getOutNodes('./seglink_2.txt')
    #checkNode_1('./original_seglink_tf_checkpoints/model.ckpt-136750') 
    freezeGraph('./original_seglink_tf_checkpoints/model.ckpt-136750', nodes, './pb_model/seglink.pb')
