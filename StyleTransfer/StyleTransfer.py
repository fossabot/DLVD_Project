import errno
import scipy.io
import scipy.misc
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np
import os

from matplotlib.pyplot import imshow

project_path = "C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer"

synset = [l.strip() for l in open(project_path + "\\model\\synset.txt").readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, between_01=False):
    # load image
    img = skimage.io.imread(path)
    #img = img / 255.0
    #assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    if between_01==False :
        return resized_img * 255.0
    else :
        return resized_img

def save_image(path, name, image, to255=False):
    # Output should add back the mean.
    #image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    if to255 == True :
        image = image * 255.0
    image = np.clip(image, 0, 255).astype('uint8')
    make_sure_path_exists(project_path + path)
    scipy.misc.imsave(project_path + path + name, image)

cat = load_image(project_path + "\\images\\cat.jpg")
elch = load_image(project_path + "\\images\\elch.jpg")
style = load_image(project_path + "\\images\\style.jpg")

# returns the top1 string
def print_prob(prob):
    # print prob
    print("prob shape", prob.shape)
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    print("Top1: ", top1)
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    print("Top5: ", top5)
    return top1


def calc_gram(single_picture_tensor_conv):
    wTimesH = int(single_picture_tensor_conv.get_shape()[0] * single_picture_tensor_conv.get_shape()[1])
    numFilters = int(single_picture_tensor_conv.get_shape()[2])
    tensor_conv_reshape = tf.reshape(single_picture_tensor_conv, (wTimesH, numFilters))
    return tf.matmul(tf.transpose(tensor_conv_reshape), tensor_conv_reshape)


def load_vgg_input( images, path = project_path + "\\model\\vgg.tfmodel"):
    print('load vgg')
    with open(path, mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    #images = tf.placeholder("float32", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images": images})
    #print("graph loaded from disk")
    print('Done')
    return tf.get_default_graph(), images


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def save_gen_weights(sess, path="", name="\\checkpoint.data"):
    print('save generator weights')
    saver = tf.train.Saver(tf.all_variables())
    make_sure_path_exists(project_path + "\\tmp" + path)
    saver.save(sess, project_path + "\\tmp" + path + name)
    print('Done')


def load_gen_weithts(sess, path=""):
    print("load generator weights")
    saver = tf.train.Saver(tf.all_variables())
    #saver = tf.train.Saver(tf.all_variables())
    c_path = project_path + "\\tmp" + path
    print(c_path)
    saver.restore(sess, tf.train.latest_checkpoint(c_path))  # now OK
    print("DONE")


def _slice_tensor(tensor):
    shape = tensorshape_to_int_array(tensor.get_shape());
    half1 = int(shape[1]/2)
    half2 = int(shape[2]/2)
    halfhalf1 = int(half1 / 2)
    halfhalf2 = int(half2 / 2)

    p1 = tf.slice(tensor, [0, 0, 0, 0], [1, half1 + halfhalf1, half2 + halfhalf2, shape[3]])
    p2 = tf.slice(tensor, [0, halfhalf1, 0, 0], [1, half1 + halfhalf1, half2 + halfhalf2, shape[3]])
    p3 = tf.slice(tensor, [0, 0, halfhalf2, 0], [1, half1 + halfhalf1, half2 + halfhalf2, shape[3]])
    p4 = tf.slice(tensor, [0, halfhalf1, halfhalf2, 0], [1, half1 + halfhalf1, half2 + halfhalf2, shape[3]])
    return p1, p2, p3, p4

def _concat_parts(p1, p2, p3, p4):
    shape = tensorshape_to_int_array(p1.get_shape())
    shape = (shape[0], int(4*shape[1]/3), int(4*shape[2]/3), shape[3])

    #assert(shape[1] == 224)
    #assert (shape[2] == 224)

    half1 = int(shape[1] / 2)
    half2 = int(shape[2] / 2)
    halfhalf1 = int(half1 / 2)
    halfhalf2 = int(half2 / 2)


    horizontal_add = tf.constant(np.zeros([1, halfhalf1, half2 + halfhalf2, shape[3]]), "float32")
    vertical_add = tf.constant(np.zeros([1, 2*half1, halfhalf2, shape[3]]), "float32")

    var = tf.constant(np.zeros(shape), dtype="float32")
    var2 = tf.add(var, tf.concat(2, [tf.concat(1, [p1, horizontal_add]), vertical_add]))
    var3 = tf.add(var2, tf.concat(2, [tf.concat(1, [horizontal_add, p2]), vertical_add]))
    var4 = tf.add(var3, tf.concat(2, [vertical_add, tf.concat(1, [p3, horizontal_add])]))
    var5 = tf.add(var4, tf.concat(2, [vertical_add, tf.concat(1, [horizontal_add, p4])]))

    a_div = np.zeros(shape, dtype="float32")
    a_div.fill(1.0)
    for i in range(half1):
        for j in range(halfhalf2):
            for k in range(shape[3]):
                a_div[0, halfhalf1 + i, j, k] = 2

    for i in range(half1):
        for j in range(halfhalf2):
            for k in range(shape[3]):
                a_div[0, halfhalf1 + i, shape[2] - 1 - j, k] = 2

    for i in range(halfhalf1):
        for j in range(half2):
            for k in range(shape[3]):
                a_div[0, i, halfhalf2 + j, k] = 2

    for i in range(halfhalf1):
        for j in range(half2):
            for k in range(shape[3]):
                a_div[0, shape[1] - 1 - i, halfhalf2 + j, k] = 2

    for i in range(half1):
        for j in range(half2):
            for k in range(shape[3]):
                a_div[0, halfhalf1 + i, halfhalf2 + j, k] = 4

    var_div = tf.constant(a_div)
    var6 = tf.div(var5, var_div)
    return var6

def build_part_layer(graph, prefix_name, part, deep, index=0):

    if deep > 1:
        p1, p2, p3, p4 = _slice_tensor(part)

        graph[prefix_name + '_L' + str(index) + '_LP1_P1'] = tf.sigmoid(_conv2d_relu(p1))
        graph[prefix_name + '_L' + str(index) + '_LP1_P2'] = tf.sigmoid(_conv2d_relu(p2))
        graph[prefix_name + '_L' + str(index) + '_LP1_P3'] = tf.sigmoid(_conv2d_relu(p3))
        graph[prefix_name + '_L' + str(index) + '_LP1_P4'] = tf.sigmoid(_conv2d_relu(p4))

        p1 = build_part_layer(graph, prefix_name, graph[prefix_name + '_L' + str(index) + '_LP1_P1'], deep - 1, index + 1)
        p2 = build_part_layer(graph, prefix_name, graph[prefix_name + '_L' + str(index) + '_LP1_P2'], deep - 1, index + 1)
        p3 = build_part_layer(graph, prefix_name, graph[prefix_name + '_L' + str(index) + '_LP1_P3'], deep - 1, index + 1)
        p4 = build_part_layer(graph, prefix_name, graph[prefix_name + '_L' + str(index) + '_LP1_P4'], deep - 1, index + 1)

        graph[prefix_name + '_L' + str(index) + '_LP2_P1'] = _concat_parts(p1, p2, p3, p4)

        return graph[prefix_name + '_L' + str(index) + '_LP2_P1']

    else:
        return part

def _relu(conv2d_layer):
    return tf.nn.relu(conv2d_layer)

variables_gen_filter = []
variables_gen_bias = []


def _conv2d(prev_layer):
    W = tf.Variable(tf.random_uniform([3,3,3,3], 0.0, 1.0, dtype='float32'), dtype="float32", name="W")
    b = tf.Variable(tf.random_uniform([3], 0.0, 1.0, dtype='float32'), dtype="float32", name="b")
    variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    return tf.add(tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME'), b)
    #return prev_layer+b


def _conv2d_relu(prev_layer):
    return _relu(_conv2d(prev_layer))


def build_gen_graph():
    graph = {}
    input_image = tf.placeholder('float32', [1, 224,224,3], name="input_image")
    graph['Generator_L0_LP1_P1'] = tf.sigmoid(_conv2d_relu(input_image))
    #graph[prefix_name + '_L' + str(index) + '_LP1_P2'] = tf.sigmoid(_conv2d_relu(graph[prefix_name + '_L' + str(index) + '_LP1_P1']))
    graph['output'] = build_part_layer(graph, 'Generator', graph['Generator_L0_LP1_P1'], 3)
    return graph, input_image


def calc_content_loss(graph, layer = "import/conv4_1/Relu:0"):
    tensor_conv = graph.get_tensor_by_name(layer)
    content_l= tf.reduce_sum(tf.square(tensor_conv[0] - tensor_conv[1]), name='content_loss')
    return content_l

def calc_gen_content_loss(gen, original):
    content_loss = tf.reduce_mean(tf.square(gen - original), name='content_loss')
    return content_loss

def tensorshape_to_int_array(ts):
    s = []
    for x in ts:
        #print(x)
        if(x.value is None):
            #print("is none")
            s.append(None)
            continue
        #print(type(x))
        s.append(int(x))
    return s


def calc_style_loss_64(graph):
    tensor_conv1_1 = graph.get_tensor_by_name("import/conv1_1/Relu:0")
    tensor_conv2_1 = graph.get_tensor_by_name("import/conv2_1/Relu:0")
    tensor_conv3_1 = graph.get_tensor_by_name("import/conv3_1/Relu:0")
    tensor_conv4_1 = graph.get_tensor_by_name("import/conv4_1/Relu:0")
    tensor_conv5_1 = graph.get_tensor_by_name("import/conv5_1/Relu:0")

    tensor_gen_gram1_1 = calc_gram(tensor_conv1_1[0])
    tensor_gen_gram2_1 = calc_gram(tensor_conv2_1[0])
    tensor_gen_gram3_1 = calc_gram(tensor_conv3_1[0])
    tensor_gen_gram4_1 = calc_gram(tensor_conv4_1[0])
    tensor_gen_gram5_1 = calc_gram(tensor_conv5_1[0])

    tensor_style_gram1_1 = calc_gram(tensor_conv1_1[2])
    tensor_style_gram2_1 = calc_gram(tensor_conv2_1[2])
    tensor_style_gram3_1 = calc_gram(tensor_conv3_1[2])
    tensor_style_gram4_1 = calc_gram(tensor_conv4_1[2])
    tensor_style_gram5_1 = calc_gram(tensor_conv5_1[2])

    s = tensorshape_to_int_array(tensor_conv1_1.get_shape())
    style_loss1_1_nominator = tf.reduce_sum(
        tf.pow(tf.cast(tensor_gen_gram1_1, tf.float64) - tf.cast(tensor_style_gram1_1, tf.float64), 2))
    style_loss1_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    style_loss1_1 = tf.div(style_loss1_1_nominator, style_loss1_1_denominator)

    s = tensorshape_to_int_array(tensor_conv2_1.get_shape())
    style_loss2_1_nominator = tf.reduce_sum(
        tf.pow(tf.cast(tensor_gen_gram2_1, tf.float64) - tf.cast(tensor_style_gram2_1, tf.float64), 2))
    style_loss2_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    style_loss2_1 = tf.div(style_loss2_1_nominator, style_loss2_1_denominator)

    s = tensorshape_to_int_array(tensor_conv3_1.get_shape())
    style_loss3_1_nominator = tf.reduce_sum(
        tf.pow(tf.cast(tensor_gen_gram3_1, tf.float64) - tf.cast(tensor_style_gram3_1, tf.float64), 2))
    style_loss3_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    style_loss3_1 = tf.div(style_loss3_1_nominator, style_loss3_1_denominator)

    s = tensorshape_to_int_array(tensor_conv4_1.get_shape())
    style_loss4_1_nominator = tf.reduce_sum(
        tf.pow(tf.cast(tensor_gen_gram4_1, tf.float64) - tf.cast(tensor_style_gram4_1, tf.float64), 2))
    style_loss4_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    style_loss4_1 = tf.div(style_loss4_1_nominator, style_loss4_1_denominator)

    s = tensorshape_to_int_array(tensor_conv5_1.get_shape())
    style_loss5_1_nominator = tf.reduce_sum(
        tf.pow(tf.cast(tensor_gen_gram5_1, tf.float64) - tf.cast(tensor_style_gram5_1, tf.float64), 2))
    style_loss5_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    style_loss5_1 = tf.div(style_loss5_1_nominator, style_loss5_1_denominator)

    style_l = style_loss1_1 + style_loss2_1 + style_loss3_1 + style_loss4_1 + style_loss5_1
    return style_l


# 0 generiert
# 1 content
# 2 style

def main():

    gen_graph, input_image = build_gen_graph()
    gen_image = gen_graph['output']

    style_image = tf.placeholder('float32', [1,224,224,3], name="style_image")
    #style_image = tf.Variable(style.reshape(1,224,224,3), trainable=False, dtype="float32", name="style_image")

    #batch = tf.nn.sigmoid(gen_image)
    batch = gen_image
    batch = tf.concat(0, [batch, input_image])
    batch = tf.concat(0, [batch, style_image])
    assert batch.get_shape() == (3, 224, 224, 3)

    graph, images = load_vgg_input(batch)

    content_loss = 0.001 * calc_content_loss(graph)
    style_loss = calc_style_loss_64(graph)
    loss = tf.cast(content_loss, tf.float64) + style_loss

    feed = {}
    feed[input_image] = cat.reshape(1,224,224,3) / 255.0
    feed[style_image] = style.reshape(1,224,224,3) / 255.0

    with tf.Session() as sess:

        # set log directory
        summary_writer = tf.train.SummaryWriter(
            project_path + '\\logs',
            graph_def=sess.graph_def)

        # 0 generiert
        # 1 content
        # 2 style

        optimizer = tf.train.AdamOptimizer()
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        variables = variables_gen_filter + variables_gen_bias
        train_step = optimizer.minimize(loss, var_list=variables)

        print(len(tf.trainable_variables()));

        init = tf.global_variables_initializer()
        sess.run(init)

        #load_gen_weithts(sess, path="\\checkStyleContent_4_plus_2")

        for i in range(4000):
            if i % 200 == 0:
                print(sess.run(loss, feed_dict=feed))
                #print(sess.run(input_image, feed_dict=feed))
                #print(sess.run(gen_image, feed_dict=feed))
                #print(sess.run(variables_gen_filter[0]))
                print(sess.run(variables_gen_bias, feed_dict=feed))
                #print(sess.run(variables[0]))
                save_image('\\output_images\\style_4_plus_5', '\\im' + str(i) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True)
                #print(sess.run(gen_graph['conv1_1'], feed_dict=feed))
            sess.run(train_step, feed_dict=feed)

        #save_image('C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\output_images\\im' + str(i) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True)
        print(sess.run(loss, feed_dict=feed))
        save_gen_weights(sess, path="\\checkStyleContent_4_plus_5")



def test():
    with tf.Session() as sess:
        init = tf.global_variables_initializer()

        val = np.zeros([1,4,4,1])
        val.fill(1.0)
        c = tf.constant(val, dtype="float64")
        print(sess.run(c))
        p1, p2, p3, p4 = _slice_tensor(c)

        x = _concat_parts(p1, p2, p3, p4)
        sess.run(init)
        print(sess.run(x))

main()
