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

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path, between_01=False, substract_mean=False):
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
        resized_img = resized_img * 255.0

    avg = 0
    if substract_mean==True:
        avg = np.sum(resized_img) / (224.0*224.0*3.0)
        resized_img -= avg

    return resized_img, avg


def save_image(path, name, image, to255=False, avg=0):
    # Output should add back the mean.
    #image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0] + avg
    if to255 == True :
        image = image * 255.0
    image = np.clip(image, 0, 255).astype('uint8')
    make_sure_path_exists(project_path + path)
    scipy.misc.imsave(project_path + path + name, image)


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

cat, avg_cat = load_image(project_path + "\\images\\cat.jpg", between_01=True, substract_mean=False)
elch, avg_elch = load_image(project_path + "\\images\\elch.jpg", between_01=True, substract_mean=False)
style, avg_style = load_image(project_path + "\\images\\style.jpg", between_01=True, substract_mean=False)
tuebingen_neckarfront, avg_tuebingen_neckarfront = load_image(project_path + "\\images\\tuebingen_neckarfront.jpg", between_01=True, substract_mean=False)


def load_vgg_input(batch, path = project_path + "\\model\\vgg.tfmodel"):
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


def load_gen_graph(path_to_graph_directory, meta_file_name):
    print('load generator')
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(project_path + path_to_graph_directory + meta_file_name)
        saver.restore(sess, tf.train.latest_checkpoint(project_path + path_to_graph_directory))
    print('Done')
    return tf.get_default_graph()


def load_gen_weithts(sess, path=""):
    print("load generator weights")
    saver = tf.train.Saver(tf.all_variables())
    #saver = tf.train.Saver(tf.all_variables())
    c_path = project_path + "\\tmp" + path
    print(c_path)
    saver.restore(sess, tf.train.latest_checkpoint(c_path))  # now OK
    print("DONE")


def save_gen_weights(sess, path="", name="\\checkpoint.data"):
    print('save generator weights')
    saver = tf.train.Saver(tf.all_variables())
    make_sure_path_exists(project_path + "\\tmp" + path)
    saver.save(sess, project_path + "\\tmp" + path + name)
    print('Done')


def export_gen_weights_android(sess, variables, path):
    for v in variables:
        with open(project_path + path + v.name, mode='w+') as f :
            shape = str(tensorshape_to_int_array(v.get_shape()));
            f.write(shape[1:(len(shape) - 1)])
            f.write('\n')
            variable = sess.run(v)
            f.write(np.reshape(variable, [1] ))


def _relu(conv2d_layer):
    return tf.nn.relu(conv2d_layer)

variables_gen_filter = []
variables_gen_bias = []


def _conv2d(prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3):
    W = tf.Variable(tf.random_uniform([filter_size, filter_size ,i_num_channel, o_num_filter], 0.0, 1.0, dtype='float32'), dtype="float32", name="W")
    b = tf.Variable(tf.random_uniform([o_num_filter], 0.0, 1.0, dtype='float32'), dtype="float32", name="b")
    variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    return tf.add(tf.nn.conv2d(prev_layer, filter=W, strides=strides, padding='SAME'), b)


def _fract_conv2d(prev_layer, strides, i_num_channel = 3, o_num_filter = 3):
    W = tf.Variable(tf.random_uniform([3,3,o_num_filter, i_num_channel], 0.0, 1.0, dtype='float32'), dtype="float32", name="W")
    b = tf.Variable(tf.random_uniform([o_num_filter], 0.0, 1.0, dtype='float32'), dtype="float32", name="b")
    variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    shape = tensorshape_to_int_array(prev_layer.get_shape())
    return tf.add(tf.nn.conv2d_transpose(prev_layer, W, [1, 2*shape[1], 2*shape[2], o_num_filter ] , strides, padding='SAME'), b)


def _conv2d_relu(prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3):
    return _relu(_conv2d(prev_layer, i_num_channel=i_num_channel, o_num_filter=o_num_filter, strides=strides, filter_size=filter_size))


def _instance_norm(x, epsilon=1e-9):
    mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
    return tf.div(tf.sub(x , mean), tf.sqrt(tf.add(var, epsilon)))



def build_gen_graph_deep():
    graph = {}
    input_image = tf.placeholder('float32', [1, 224, 224, 3], name="ph_input_image")
    #graph['var_input_image'] = _fract_pooling_downsample(input_image)
    #print(graph['var_input_image'].get_shape())
    graph['conv1_0'] = _instance_norm(_conv2d(input_image, filter_size=9, o_num_filter=32))
    print(graph['conv1_0'].get_shape())

    graph['conv2_0'] = _instance_norm(_conv2d(graph['conv1_0'], strides=[1, 2, 2, 1], i_num_channel = 32, o_num_filter = 64))
    print(graph['conv2_0'].get_shape())
    graph['conv2_1'] = _instance_norm(_conv2d(graph['conv2_0'], strides=[1, 2, 2, 1], i_num_channel=64, o_num_filter=128))
    print(graph['conv2_1'].get_shape())

    graph['conv3_0_0'] = _relu(_instance_norm(_conv2d(graph['conv2_1'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_0_1'] = _relu(_instance_norm(_conv2d(graph['conv3_0_0'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_1_0'] = _relu(_instance_norm(_conv2d(graph['conv3_0_1'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_1_1'] = _relu(_instance_norm(_conv2d(graph['conv3_1_0'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_2_0'] = _relu(_instance_norm(_conv2d(graph['conv3_1_1'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_2_1'] = _relu(_instance_norm(_conv2d(graph['conv3_2_0'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_3_0'] = _relu(_instance_norm(_conv2d(graph['conv3_2_1'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_3_1'] = _relu(_instance_norm(_conv2d(graph['conv3_3_0'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_4_0'] = _relu(_instance_norm(_conv2d(graph['conv3_3_1'], i_num_channel=128, o_num_filter=128)))
    graph['conv3_4_1'] = _relu(_instance_norm(_conv2d(graph['conv3_4_0'], i_num_channel=128, o_num_filter=128)))
    print(graph['conv3_4_1'].get_shape())

    graph['conv4_0'] = _instance_norm(_fract_conv2d(graph['conv3_4_1'], [1, 2, 2, 1], i_num_channel=128, o_num_filter=64))
    print(graph['conv4_0'].get_shape())
    graph['conv4_1'] = _instance_norm(_fract_conv2d(graph['conv4_0'], [1, 2, 2, 1], i_num_channel=64, o_num_filter=32))
    print(graph['conv4_1'].get_shape())

    graph['conv5_0'] = _instance_norm(_conv2d(graph['conv4_1'], i_num_channel=32, o_num_filter=3, filter_size=9))

    graph['output'] = tf.tanh(graph['conv5_0'], 'output')
    return graph, input_image


def calc_gram(single_picture_tensor_conv):
    wTimesH = int(single_picture_tensor_conv.get_shape()[0] * single_picture_tensor_conv.get_shape()[1])
    numFilters = int(single_picture_tensor_conv.get_shape()[2])
    tensor_conv_reshape = tf.reshape(single_picture_tensor_conv, (wTimesH, numFilters))
    return tf.matmul(tf.transpose(tensor_conv_reshape), tensor_conv_reshape)


def calc_content_loss(graph, layer = "import/conv3_3/Relu:0"):
    tensor_conv = graph.get_tensor_by_name(layer)
    content_l= tf.reduce_sum(tf.square(tensor_conv[0] - tensor_conv[1]), name='content_loss')
    return content_l


def calc_gen_content_loss(gen, original):
    content_loss = tf.reduce_mean(tf.square(gen - original), name='content_loss')
    return content_loss


def calc_style_loss_64(graph):
    tensor_conv1_1 = graph.get_tensor_by_name("import/conv1_2/Relu:0")
    tensor_conv2_1 = graph.get_tensor_by_name("import/conv2_2/Relu:0")
    tensor_conv3_1 = graph.get_tensor_by_name("import/conv3_2/Relu:0")
    tensor_conv4_1 = graph.get_tensor_by_name("import/conv4_2/Relu:0")
    #tensor_conv5_1 = graph.get_tensor_by_name("import/conv5_2/Relu:0")

    tensor_gen_gram1_1 = calc_gram(tensor_conv1_1[0])
    tensor_gen_gram2_1 = calc_gram(tensor_conv2_1[0])
    tensor_gen_gram3_1 = calc_gram(tensor_conv3_1[0])
    tensor_gen_gram4_1 = calc_gram(tensor_conv4_1[0])
    #tensor_gen_gram5_1 = calc_gram(tensor_conv5_1[0])

    tensor_style_gram1_1 = calc_gram(tensor_conv1_1[2])
    tensor_style_gram2_1 = calc_gram(tensor_conv2_1[2])
    tensor_style_gram3_1 = calc_gram(tensor_conv3_1[2])
    tensor_style_gram4_1 = calc_gram(tensor_conv4_1[2])
    #tensor_style_gram5_1 = calc_gram(tensor_conv5_1[2])

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

    #s = tensorshape_to_int_array(tensor_conv5_1.get_shape())
    #style_loss5_1_nominator = tf.reduce_sum(
    #    tf.pow(tf.cast(tensor_gen_gram5_1, tf.float64) - tf.cast(tensor_style_gram5_1, tf.float64), 2))
    #style_loss5_1_denominator = tf.cast(4.0 * (s[1] * s[2]) ** 2 * s[3] ** 2.0, tf.float64)
    #style_loss5_1 = tf.div(style_loss5_1_nominator, style_loss5_1_denominator)

    style_l = style_loss1_1 + style_loss2_1 + style_loss3_1 + style_loss4_1
    return style_l


# 0 generiert
# 1 content
# 2 style

def main():

    gen_graph, input_image = build_gen_graph_deep()
    gen_image = gen_graph['output']

    style_image = tf.placeholder('float32', [1, 224, 224,3], name="style_image")

    batch = gen_image
    batch = tf.concat(0, [batch, input_image])
    batch = tf.concat(0, [batch, style_image])
    assert batch.get_shape() == (3, 224, 224, 3)

    graph = load_vgg_input(batch)

    content_loss = 0.001 * calc_content_loss(graph)
    style_loss = calc_style_loss_64(graph)
    loss = tf.cast(content_loss, tf.float64) + style_loss

    feed = {}
    feed[input_image] = tuebingen_neckarfront.reshape(1, 224, 224,3)
    feed[style_image] = style.reshape(1, 224, 224,3)


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
        sess.run(init, feed)

        #load_gen_weithts(sess, path="\\checkStyleContent_34_plus_34_k")

        i = 0
        for i in range(20000):
            print(i)
            if i % 250 == 0:
                save_image('\\output_images\\style_20_plus_37_k', '\\im' + str(i) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True, avg=avg_tuebingen_neckarfront)
                #print(sess.run(gen_graph['conv1_1'], feed_dict=feed))
            sess.run(train_step, feed_dict=feed)

        save_image('\\output_images\\style_20_plus_37_k', '\\im' + str(i+1) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True, avg=avg_tuebingen_neckarfront)
        print(sess.run(loss, feed_dict=feed))
        save_gen_weights(sess, path="\\checkStyleContent_20_plus_37_k")
        #export_gen_weights_android(sess, variables, '\\test')


def transform(image, path_to_generator, meta_filename, save_to_directory, filename):
    graph = load_gen_graph(path_to_generator, meta_filename)
    feed = {'ph_input_image:0' : image}

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init, feed_dict=feed)
        x = sess.run(graph.get_tensor_by_name('output:0'), feed_dict=feed)
        make_sure_path_exists(project_path + '\\output_images' + save_to_directory)
        save_image('\\output_images' + save_to_directory, filename, x, True)


main()
#transform(np.reshape(elch, [1,224,224,3]), '\\tmp\\checkStyleContent_10_plus_12', '\\checkpoint.data.meta', '\\style_10_plus_12', '\\elch.jpg')