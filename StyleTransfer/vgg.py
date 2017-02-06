import config as conf
import numpy as np

def load_vgg_input(tf, batch, path = conf.project_path + conf.model_path + "\\vgg.tfmodel"):
    print('load vgg')
    with open(path, mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    avg_custom = np.array([123.68 / 255.0, 116.779 / 255.0, 103.939 / 255.0]).reshape([1, 1, 1, 3])
    vgg_average = tf.constant(avg_custom, dtype="float32")
    tf.import_graph_def(graph_def, input_map={"images": tf.div(batch, vgg_average)})
    #print("graph loaded from disk")
    print('Done')

    return tf.get_default_graph()