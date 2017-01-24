import errno
import scipy.io
import scipy.misc
import tensorflow as tf
from tensorflow.contrib.copy_graph import copy_variable_to_graph
import skimage
import skimage.io
import skimage.transform
import numpy as np
import os


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
def load_image(path, between_01=False, substract_mean=False, output_size=224):
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
    resized_img = skimage.transform.resize(crop_img, (output_size, output_size))
    if between_01==False :
        resized_img = resized_img * 255.0

    avg = 0
    if substract_mean==True:
        avg = np.sum(resized_img) / (output_size*output_size*3.0)
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


def create_saver(sess) :
    print("create saver")
    saver = tf.train.Saver(tf.all_variables())
    print('Done')
    return saver


def load_gen_last_checkpoint(sess, saver,  path=""):
    print("load last checkpoint")
    c_path = project_path + "\\tmp" + path
    print(c_path)
    saver.restore(sess, tf.train.latest_checkpoint(c_path))
    print("DONE")


def save_gen_checkpoint(sess, saver, path="", name="\\checkpoint.data"):
    print('save checkpoint')
    make_sure_path_exists(project_path + "\\tmp" + path)
    saver.save(sess, project_path + "\\tmp" + path + name)
    print('Done')


def export_gen_weights_android(sess, variables, path):
    for v in variables:
        make_sure_path_exists(project_path + path)

        f = open(project_path + path + "\\" + str(v.name)[0:(len(str(v.name))-2)], 'w')
        int_shape = tensorshape_to_int_array(v.get_shape())
        shape = str(int_shape)
        f.write(shape[1:(len(shape) - 1)])
        f.write("\n")
        variable = sess.run(v)
        to_write = np.reshape(variable, np.prod(int_shape))
        for a in to_write:
            f.write(str(a))
            f.write("\n")
        f.close()


def export_gen_graph(sess, variables_filter, variables_bias, path, name="gen_export.pb") :

    var_gen_filter_new = []
    for i in range(len(variables_filter)):
        var_gen_filter_new.append(sess.run(variables_filter[i]))

    var_gen_bias_new = []
    for i in range(len(variables_bias)):
        var_gen_bias_new.append(sess.run(variables_bias[i]))

    to_graph = tf.Graph()
    with to_graph.as_default() as g:
        graph, image, var_gen_filter, var_gen_bias = build_gen_graph_deep()

        #saver = tf.train.Saver(tf.all_variables())
        make_sure_path_exists(project_path + "\\android_exports" + path)
        with tf.Session() as new_sess:
            init = tf.global_variables_initializer()
            new_sess.run(init)
            #summary_writer = tf.train.SummaryWriter(project_path + '\\android_exports\\logs', graph_def=new_sess.graph_def)

            for i in range(len(var_gen_filter)) :
                new_sess.run(tf.assign(var_gen_filter[i], var_gen_filter_new[i]))
            for i in range(len(var_gen_bias)):
                new_sess.run(tf.assign(var_gen_bias[i], var_gen_bias_new[i]))

            #saver.save(new_sess, project_path + "\\android_exports" + path + name)
            tf.train.write_graph(tf.get_default_graph(), project_path + "\\android_exports" + path, name, as_text=False)



def _relu(conv2d_layer):
    return tf.nn.relu(conv2d_layer)



def _weight_loss(variables_gen_filter, variables_gen_bias):
    _w_loss = tf.reduce_sum(tf.square(variables_gen_filter[0]))
    for w in variables_gen_filter[1:]:
        _w_loss = tf.add(tf.reduce_sum(tf.square(w)), _w_loss)

    _b_loss = tf.reduce_sum(tf.square(variables_gen_bias[0]))
    for b in variables_gen_bias[1:]:
        _b_loss = tf.add(tf.reduce_sum(tf.square(b)), _b_loss)

    return tf.add(_w_loss, _b_loss)



def _conv2d(variables_gen_filter, variables_gen_bias, prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3, pad='SAME'):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    W = tf.Variable(tf.random_normal([filter_size, filter_size ,i_num_channel, o_num_filter], dtype='float32', stddev=var), dtype="float32", name="W")
    b = tf.Variable(tf.random_normal([o_num_filter], stddev=0.1, dtype='float32'), dtype="float32", name="b")
    variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    return tf.add(tf.nn.conv2d(prev_layer, filter=W, strides=strides, padding=pad), b)


def _fract_conv2d(variables_gen_filter, variables_gen_bias, prev_layer, strides, i_num_channel = 3, o_num_filter = 3, pad='SAME', filter_size=3):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    W = tf.Variable(tf.random_normal([filter_size,filter_size,o_num_filter, i_num_channel], dtype='float32', stddev=var), dtype="float32", name="W")
    b = tf.Variable(tf.random_normal([o_num_filter], dtype='float32', stddev=0.1), dtype="float32", name="b")
    variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    shape = tensorshape_to_int_array(prev_layer.get_shape())
    return tf.add(tf.nn.conv2d_transpose(prev_layer, W, [1, 2*shape[1], 2*shape[2], o_num_filter ] , strides, padding=pad), b)


def _conv2d_relu(variables_gen_filter, variables_gen_bias, prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3):
    return _relu(_conv2d(variables_gen_filter, variables_gen_bias, prev_layer, i_num_channel=i_num_channel, o_num_filter=o_num_filter, strides=strides, filter_size=filter_size))


def _instance_norm(x, epsilon=1e-9):
    mean, var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
    return tf.div(tf.sub(x , mean), tf.sqrt(tf.add(var, epsilon)))


def _clip_2x2_border(x):
    shape = tensorshape_to_int_array(x.get_shape())
    height_end = shape[1] - 4
    with_end = shape[2] - 4
    tmp = tf.slice(x, [0, 2, 2, 0], [shape[0], height_end, with_end, shape[3]])
    return tmp


def build_gen_graph_deep():

    variables_gen_filter = []
    variables_gen_bias = []

    graph = {}
    input_image = tf.placeholder('float32', [1, 304, 304, 3], name="ph_input_image")

    graph['conv1_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, input_image, filter_size=9, o_num_filter=32)))
    print(graph['conv1_0'].get_shape())

    graph['conv2_0'] = _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv1_0'], strides=[1, 2, 2, 1], i_num_channel = 32, o_num_filter = 64))
    print(graph['conv2_0'].get_shape())
    graph['conv2_1'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv2_0'], strides=[1, 2, 2, 1], i_num_channel=64, o_num_filter=128)))
    print(graph['conv2_1'].get_shape())

    graph['conv3_0_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv2_1'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_0_1'] = tf.add(_clip_2x2_border(graph['conv2_1']), _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_0_0'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_1_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_0_1'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_1_1'] = tf.add(_clip_2x2_border(graph['conv3_0_1']), _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_1_0'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_2_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_1_1'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_2_1'] = tf.add(_clip_2x2_border(graph['conv3_1_1']), _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_2_0'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_3_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_2_1'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_3_1'] = tf.add(_clip_2x2_border(graph['conv3_2_1']), _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_3_0'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_4_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_3_1'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    graph['conv3_4_1'] = tf.add(_clip_2x2_border(graph['conv3_3_1']), _instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_4_0'], i_num_channel=128, o_num_filter=128, pad='VALID')))
    print(graph['conv3_4_1'].get_shape())

    graph['conv4_0'] = _instance_norm(_fract_conv2d(variables_gen_filter, variables_gen_bias, graph['conv3_4_1'], [1, 2, 2, 1], i_num_channel=128, o_num_filter=64))
    print(graph['conv4_0'].get_shape())
    graph['conv4_1'] = _relu(_instance_norm(_fract_conv2d(variables_gen_filter, variables_gen_bias, graph['conv4_0'], [1, 2, 2, 1], i_num_channel=64, o_num_filter=32)))
    print(graph['conv4_1'].get_shape())

    graph['conv5_0'] = _relu(_instance_norm(_conv2d(variables_gen_filter, variables_gen_bias, graph['conv4_1'], i_num_channel=32, o_num_filter=3, filter_size=9)))

    graph['output'] = tf.tanh(graph['conv5_0'], 'output')
    return graph, input_image, variables_gen_filter, variables_gen_bias


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
        tf.pow(tensor_gen_gram1_1 - tensor_style_gram1_1, 2.0))
    style_loss1_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
    style_loss1_1 = tf.div(style_loss1_1_nominator, style_loss1_1_denominator)

    s = tensorshape_to_int_array(tensor_conv2_1.get_shape())
    style_loss2_1_nominator = tf.reduce_sum(
        tf.pow(tensor_gen_gram2_1 - tensor_style_gram2_1, 2.0))
    style_loss2_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
    style_loss2_1 = tf.div(style_loss2_1_nominator, style_loss2_1_denominator)

    s = tensorshape_to_int_array(tensor_conv3_1.get_shape())
    style_loss3_1_nominator = tf.reduce_sum(
        tf.pow(tensor_gen_gram3_1 - tensor_style_gram3_1, 2.0))
    style_loss3_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
    style_loss3_1 = tf.div(style_loss3_1_nominator, style_loss3_1_denominator)

    s = tensorshape_to_int_array(tensor_conv4_1.get_shape())
    style_loss4_1_nominator = tf.reduce_sum(
        tf.pow(tensor_gen_gram4_1 - tensor_style_gram4_1, 2.0))
    style_loss4_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
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

    cat, avg_cat = load_image(project_path + "\\images\\cat.jpg", between_01=True, substract_mean=False)
    cat_gen, avg_cat_gen = load_image(project_path + "\\images\\cat.jpg", between_01=True, substract_mean=False, output_size=304)

    elch, avg_elch = load_image(project_path + "\\images\\elch.jpg", between_01=True, substract_mean=False)

    style_red, avg_style_red = load_image(project_path + "\\images\\style.jpg", between_01=True, substract_mean=False)
    style_blue, avg_style_blue = load_image(project_path + "\\images\\style2.jpg", between_01=True, substract_mean=False)
    tuebingen_neckarfront, avg_tuebingen_neckarfront = load_image(project_path + "\\images\\tuebingen_neckarfront.jpg", between_01=True, substract_mean=False)
    tuebingen_neckarfront_gen, avg_tuebingen_neckarfront_gen = load_image(project_path + "\\images\\tuebingen_neckarfront.jpg", between_01=True, substract_mean=False, output_size=304)


    gen_graph, input_image, variables_gen_filter, variables_gen_bias = build_gen_graph_deep()
    gen_image = gen_graph['output']

    style_image = tf.placeholder('float32', [1, 224, 224,3], name="style_image")
    content_input = tf.placeholder('float32', [1, 224, 224,3], name="content_image")

    batch = gen_image
    batch = tf.concat(0, [batch, content_input])
    batch = tf.concat(0, [batch, style_image])
    assert batch.get_shape() == (3, 224, 224, 3)

    graph = load_vgg_input(batch)

    content_loss = 0.001 * calc_content_loss(graph)
    style_loss = calc_style_loss_64(graph)
    weight_loss = 10.0 * _weight_loss(variables_gen_filter, variables_gen_bias)
    loss = content_loss + style_loss + weight_loss

    learning_rate = 0.001;
    var_learning_rate = tf.placeholder("float32")

    feed = {}
    feed[input_image] = cat_gen.reshape(1, 304, 304, 3)
    feed[content_input] = cat.reshape(1, 224, 224, 3)
    feed[style_image] = style_red.reshape(1, 224, 224,3)
    feed[var_learning_rate] = learning_rate;

    graph.as_default()

    with tf.Session() as sess:

        # set log directory
        summary_writer = tf.train.SummaryWriter(
            project_path + '\\logs',
            graph_def=sess.graph_def)

        # 0 generiert
        # 1 content
        # 2 style


        #optimizer = tf.train.MomentumOptimizer(0.0001, 0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=var_learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        variables = variables_gen_filter + variables_gen_bias
        train_step = optimizer.minimize(loss, var_list=variables)

        print(len(tf.trainable_variables()));

        init = tf.global_variables_initializer()
        sess.run(init, feed)


        loading_directory = "\\checkStyleContent_15_plus_43_k"
        saving_directory = "\\checkStyleContent_20_plus_43_k"
        starting_pic_num = 0

        saver = create_saver(sess)
        load_gen_last_checkpoint(sess, saver, path=loading_directory)


        i = 0
        last_l = sess.run(loss, feed_dict=feed)
        last_cl = sess.run(content_loss, feed_dict=feed)
        last_sl = sess.run(style_loss, feed_dict=feed)
        last_wl = sess.run(weight_loss, feed_dict=feed)

        neg_loss_counter = 0
        restore= False
        for i in range(2):
            print(i)
            if i % 250 == 0:
                l = sess.run(loss, feed_dict=feed)

                if (last_l -l ) < 0 and i != 0:
                    neg_loss_counter += 1
                    print('neg loss -> counter increase :' + str(neg_loss_counter))
                    if neg_loss_counter >= 1 :
                        learning_rate /= 2
                        neg_loss_counter = 0
                        restore = True
                        print("new learning rate : " + str(learning_rate))

                print('loss : ' + str(l))
                print('loss_improvement : ' + str((last_l - l) / last_l))
                last_l = l

                cl = sess.run(content_loss, feed_dict=feed)
                print('content_loss : ' + str(cl))
                print('content_loss_improvement : ' + str((last_cl - cl) / last_cl))
                last_cl=cl
                sl = sess.run(style_loss, feed_dict=feed)
                print('style_loss : ' + str(sl))
                print('style_loss_improvement : ' + str((last_sl - sl) / last_sl))
                last_sl=sl
                wl = sess.run(weight_loss, feed_dict=feed)
                print('weight_loss : ' + str(wl))
                print('weight_loss_improvement : ' + str((last_wl - wl) / last_wl))
                last_wl=wl
                save_image('\\output_images' + saving_directory, '\\im' + str(i + starting_pic_num) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True, avg=avg_tuebingen_neckarfront)

                if restore == False :
                    save_gen_checkpoint(sess, saver, path=saving_directory)
                else :
                    print("Restoring last checkpoint")
                    load_gen_last_checkpoint(sess, saver, path=saving_directory)
                    restore = False
                    print("Done")

            sess.run(train_step, feed_dict=feed)

        save_image('\\output_images' + saving_directory, '\\im' + str(i + starting_pic_num + 1) + '.jpg', sess.run(gen_image, feed_dict=feed), to255=True, avg=avg_tuebingen_neckarfront)
        print(sess.run(loss, feed_dict=feed))
        save_gen_checkpoint(sess, saver, path=saving_directory)
        #export_gen_graph(sess, variables_gen_filter, variables_gen_bias, saving_directory)


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
