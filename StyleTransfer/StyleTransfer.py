import errno
import scipy.io
import scipy.misc
import tensorflow as tf
import time
import skimage
import skimage.io
import skimage.transform
import numpy as np
import os


project_path = "C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer"

model_path = "\\data\\model"
checkpoints_path = "\\data\\checkpoints"
images_path = "\\data\\images"

log_train = "\\logs\\training_network"
log_generator = "\\logs\\generator_network"


output_generator = "\\outputs\\generator_networks"
output_images = "\\outputs\\images"

VGG_STYLE_TENSOR_1 = "import/conv1_2/Relu:0"
VGG_STYLE_TENSOR_2 = "import/conv2_2/Relu:0"
VGG_STYLE_TENSOR_3 = "import/conv3_2/Relu:0"
VGG_STYLE_TENSOR_4 = "import/conv4_2/Relu:0"

VGG_CONTENT_LAYER = "import/conv1_2/Relu:0"

BATCH_SIZE = 4
PRECOMPUTE_BATCH_SIZE = 20

def time_to_str(time):
    t_in_min = time / 60.0
    t_in_hours = time / (60.0*60.0)

    frac_min = int((t_in_hours - float(int(t_in_hours)))*60.0)
    frac_sec = int((t_in_min - float(int(t_in_min))) * 60.0)
    return str(int(t_in_hours)) + 'h ' + str(frac_min) + 'm ' + str(frac_sec) + 's'


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
    img = skimage.io.imread(project_path + images_path + path)
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


def save_image(path, name, images, to255=False, avg=0):
    # Output should add back the mean.
    #image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    for i in range(len(images)):
        image = images[i] + avg
        if to255 == True :
            image = image * 255.0
        image = np.clip(image, 0, 255).astype('uint8')
        full_path = project_path + output_images + path
        if len(images) != 1:
            full_path = project_path + output_images + path + '\\' + str(i)
        make_sure_path_exists(full_path)
        scipy.misc.imsave(full_path + name + '_' + str(i) + '.jpg', image)


def load_pictures_for_feed(directory_path, recursive=False):
    print("Loading pictures : " + directory_path)
    images = []
    content_images = []
    for file in os.listdir(project_path + images_path + directory_path):
        full_path = os.path.join(project_path + images_path + directory_path, file)
        if os.path.isfile(full_path) and str(file)[-4:] == '.jpg' :
            img, avg_img = load_image(directory_path + '\\' + str(file), between_01=True, output_size=234)

            if len(img.shape) < 3 or img.shape[2] != 3 :
                print("Picture with less than 3 channels found : " + str(file) + "\t Picture will be skipped.")
                continue

            images.append(img)
            img, avg_img = load_image(directory_path + '\\' + str(file), between_01=True)
            content_images.append(img)
        else:
            if recursive and not os.path.isfile(full_path):
                im, con = load_pictures_for_feed(directory_path + '\\' + file)
                for i in im:
                    images.append(i)
                for c in con:
                    content_images.append(c)

    print("Done. Number of pictures loaded : " + str(len(images)))
    return images, content_images

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

def load_vgg_input(batch, path = project_path + model_path + "\\vgg.tfmodel"):
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
    saver = tf.train.Saver(tf.global_variables())
    print('Done')
    return saver


def load_gen_last_checkpoint(sess, saver,  path=""):
    print("load last checkpoint")
    c_path = project_path + checkpoints_path + path
    print(c_path)
    saver.restore(sess, tf.train.latest_checkpoint(c_path))
    print("DONE")


def save_gen_checkpoint(sess, saver, path="", name="\\checkpoint.data"):
    print('save checkpoint')
    make_sure_path_exists(project_path + checkpoints_path + path)
    saver.save(sess, project_path + checkpoints_path + path + name)
    print('Done')


def export_gen_graph(sess, variables_filter, variables_bias, variables_scalars, path, name="gen_export_540.pb") :

    var_gen_filter_new = []
    for i in range(len(variables_filter)):
        var_gen_filter_new.append(sess.run(variables_filter[i]))

    var_gen_bias_new = []
    for i in range(len(variables_bias)):
        var_gen_bias_new.append(sess.run(variables_bias[i]))

    var_gen_scalars_new = []
    for i in range(len(variables_scalars)):
        var_gen_scalars_new.append(sess.run(variables_scalars[i]))

    to_graph = tf.Graph()
    with to_graph.as_default() as g:
        build_gen_graph_deep(trainable=False, variables_gen_filter=var_gen_filter_new, variables_gen_bias=var_gen_bias_new, variables_scalars=var_gen_scalars_new, input_resolution=540)

        #saver = tf.train.Saver(tf.all_variables())
        make_sure_path_exists(project_path + output_generator + path)
        with tf.Session() as new_sess:
            init = tf.global_variables_initializer()
            new_sess.run(init)
            #summary_writer = tf.train.SummaryWriter(project_path + log_generator, graph_def=new_sess.graph_def)

            #saver.save(new_sess, project_path + "\\android_exports" + path + name)
            tf.train.write_graph(tf.get_default_graph(), project_path + output_generator + path, name, as_text=False)



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



def _conv2d(variables_gen_filter, variables_gen_bias, prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3, pad='SAME', is_trainable = True):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    if is_trainable:
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size ,i_num_channel, o_num_filter], dtype='float32', stddev=var), dtype="float32", name="W", trainable=is_trainable)
        b = tf.Variable(tf.truncated_normal([o_num_filter], stddev=0.1, dtype='float32'), dtype="float32", name="b", trainable=is_trainable)
        variables_gen_filter.append(W)
        variables_gen_bias.append(b)
    else:
        W = tf.constant(variables_gen_filter.pop(0), dtype="float32", name="W")
        b = tf.constant(variables_gen_bias.pop(0), dtype="float32", name="b")
    return tf.add(tf.nn.conv2d(prev_layer, filter=W, strides=strides, padding=pad), b)


def _fract_conv2d(variables_gen_filter, variables_gen_bias, prev_layer, strides, i_num_channel = 3, o_num_filter = 3, pad='SAME', filter_size=3, is_trainable = True):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    if is_trainable:
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,o_num_filter, i_num_channel], dtype='float32', stddev=var), dtype="float32", name="W", trainable=is_trainable)
        b = tf.Variable(tf.truncated_normal([o_num_filter], dtype='float32', stddev=0.1), dtype="float32", name="b", trainable=is_trainable)
        variables_gen_filter.append(W)
        variables_gen_bias.append(b)
    else:
        W = tf.constant(variables_gen_filter.pop(0), dtype="float32", name="W")
        b = tf.constant(variables_gen_bias.pop(0), dtype="float32", name="b")
    shape = tensorshape_to_int_array(prev_layer.get_shape())
    return tf.add(tf.nn.conv2d_transpose(prev_layer, W, [shape[0], strides[1]*shape[1], strides[2]*shape[2], o_num_filter ] , strides, padding=pad), b)


def _instance_norm(variable_scalars, x, epsilon=1e-3, is_trainable=True):
    batch, rows, cols, channels = [i.value for i in x.get_shape()]
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    if is_trainable:
        s = tf.Variable(tf.ones([channels]), dtype="float32", name="s", trainable=is_trainable)
        z = tf.Variable(tf.zeros([channels]) , dtype="float32", name="z", trainable=is_trainable)
        variable_scalars.append(s)
        variable_scalars.append(z)
    else:
        s = tf.constant(variable_scalars.pop(0), dtype="float32", name="s")
        z = tf.constant(variable_scalars.pop(0), dtype="float32", name="z")
    return s * tf.div(tf.sub(x , mean), tf.sqrt(tf.add(var, epsilon))) + z


def _clip_2x2_border(x):
    shape = tensorshape_to_int_array(x.get_shape())
    height_end = shape[1] - 4
    with_end = shape[2] - 4
    tmp = tf.slice(x, [0, 2, 2, 0], [shape[0], height_end, with_end, shape[3]])
    return tmp


def build_gen_graph_deep(trainable = True, variables_gen_filter = [], variables_gen_bias = [], variables_scalars = [], input_pictures = 1, input_resolution = 234):

    if trainable :
        variables_gen_filter = []
        variables_gen_bias = []
        variables_scalars = []

    graph = {}
    input_image = tf.placeholder('float32', [input_pictures, input_resolution, input_resolution, 3], name="ph_input_image")

    graph['conv1_0'] = _relu(
                        _instance_norm(
                            variables_scalars,
                            _conv2d(
                                variables_gen_filter,
                                variables_gen_bias,
                                input_image,
                                filter_size=9,
                                o_num_filter=32,
                                is_trainable = trainable),
                            is_trainable=trainable))
    print(graph['conv1_0'].get_shape())

    graph['conv2_0'] = _instance_norm(
                        variables_scalars,
                        _conv2d(
                            variables_gen_filter,
                            variables_gen_bias,
                            graph['conv1_0'],
                            strides=[1, 3, 3, 1],
                            i_num_channel = 32,
                            o_num_filter = 64,
                            is_trainable = trainable
                        ),
                        is_trainable=trainable
    )
    print(graph['conv2_0'].get_shape())

    graph['conv2_1'] = _relu(
                        _instance_norm(
                            variables_scalars,
                            _conv2d(
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv2_0'],
                                strides=[1, 3, 3, 1],
                                i_num_channel=64,
                                o_num_filter=128,
                                is_trainable = trainable
                            ),
                        is_trainable=trainable
                        )
    )
    print(graph['conv2_1'].get_shape())

    graph['conv3_0_0'] = _relu(
                            _instance_norm(
                                variables_scalars,
                                _conv2d(variables_gen_filter, variables_gen_bias, graph['conv2_1'],
                                        i_num_channel=128,
                                        o_num_filter=128,
                                        pad='SAME',
                                        is_trainable = trainable
                                        ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_0_1'] = tf.add(
                            graph['conv2_1'],
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_0_0'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_1_0'] = _relu(
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_0_1'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_1_1'] = tf.add(
                            graph['conv3_0_1'],
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_1_0'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_2_0'] = _relu(
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_1_1'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_2_1'] = tf.add(graph['conv3_1_1'],
                                _instance_norm(
                                    variables_scalars,
                                    _conv2d(
                                        variables_gen_filter,
                                        variables_gen_bias,
                                        graph['conv3_2_0'],
                                        i_num_channel=128,
                                        o_num_filter=128,
                                        pad='SAME',
                                        is_trainable = trainable
                                    ),
                                is_trainable=trainable
                                )
    )

    graph['conv3_3_0'] = _relu(
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_2_1'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_3_1'] = tf.add(graph['conv3_2_1'],
                                _instance_norm(
                                    variables_scalars,
                                    _conv2d(
                                        variables_gen_filter,
                                        variables_gen_bias,
                                        graph['conv3_3_0'],
                                        i_num_channel=128,
                                        o_num_filter=128,
                                        pad='SAME',
                                        is_trainable = trainable
                                    ),
                                    is_trainable=trainable
                                )
    )

    graph['conv3_4_0'] = _relu(
                            _instance_norm(
                                variables_scalars,
                                _conv2d(
                                    variables_gen_filter,
                                    variables_gen_bias,
                                    graph['conv3_3_1'],
                                    i_num_channel=128,
                                    o_num_filter=128,
                                    pad='SAME',
                                    is_trainable = trainable
                                ),
                                is_trainable=trainable
                            )
    )

    graph['conv3_4_1'] = tf.add(graph['conv3_3_1'],
                                _instance_norm(
                                    variables_scalars,
                                    _conv2d(
                                        variables_gen_filter,
                                        variables_gen_bias,
                                        graph['conv3_4_0'],
                                        i_num_channel=128,
                                        o_num_filter=128,
                                        pad='SAME',
                                        is_trainable = trainable
                                    ),
                                    is_trainable=trainable
                                )
    )
    print(graph['conv3_4_1'].get_shape())

    graph['conv4_0'] = _instance_norm(
                        variables_scalars,
                        _fract_conv2d(
                            variables_gen_filter,
                            variables_gen_bias,
                            graph['conv3_4_1'],
                            [1, 3, 3, 1],
                            i_num_channel=128,
                            o_num_filter=64,
                            is_trainable = trainable
                        ),
                        is_trainable=trainable
    )
    print(graph['conv4_0'].get_shape())

    graph['conv4_1'] = _relu(
                        _instance_norm(
                            variables_scalars,
                            _fract_conv2d(
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv4_0'],
                                [1, 3, 3, 1],
                                i_num_channel=64,
                                o_num_filter=32,
                                is_trainable = trainable
                            ),
                            is_trainable=trainable
                        )
    )
    print(graph['conv4_1'].get_shape())

    graph['conv5_0'] = _relu(
                        _instance_norm(
                            variables_scalars,
                            _conv2d(
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv4_1'],
                                i_num_channel=32,
                                o_num_filter=3,
                                filter_size=9,
                                is_trainable = trainable
                            ),
                            is_trainable=trainable
                        )
    )

    graph['output'] = tf.tanh(graph['conv5_0'], name='output')

    return graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars


def precompute_style_gram(style_image, content_images):
    print("Precompute style tensors")
    graph = tf.Graph()
    with graph.as_default() as g:
        inp = tf.placeholder("float32", [None, 224, 224, 3])
        load_vgg_input(inp)

        tensor_conv1_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_1)
        tensor_conv2_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_2)
        tensor_conv3_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_3)
        tensor_conv4_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_4)

        tensor_style_gram1_1 = calc_gram(tensor_conv1_1[0])
        tensor_style_gram2_1 = calc_gram(tensor_conv2_1[0])
        tensor_style_gram3_1 = calc_gram(tensor_conv3_1[0])
        tensor_style_gram4_1 = calc_gram(tensor_conv4_1[0])

        feed = {}
        feed[inp] = style_image.reshape(1, 224, 224, 3)
        with tf.Session() as sess :
            gram_1 = sess.run(tensor_style_gram1_1, feed_dict=feed)
            gram_2 = sess.run(tensor_style_gram2_1, feed_dict=feed)
            gram_3 = sess.run(tensor_style_gram3_1, feed_dict=feed)
            gram_4 = sess.run(tensor_style_gram4_1, feed_dict=feed)

        tensor_conv = graph.get_tensor_by_name(VGG_CONTENT_LAYER)

        feed = {}
        content = []
        counter = 0
        batch_size = PRECOMPUTE_BATCH_SIZE
        brek_now = False
        while True :
            if counter % 10 == 0:
                print("Number of pictures already computed : " + str(counter))

            if counter + batch_size > len(content_images):
                batch_size = len(content_images) - counter
                brek_now = True

            feed[inp] = content_images[counter : counter + batch_size]

            counter += batch_size
            with tf.Session() as sess:
                x = sess.run(tensor_conv, feed_dict=feed)
                for i in range(len(x)):
                    content.append(x[i])

            if brek_now:
                break;

    print("Done")
    return [gram_1, gram_2, gram_3, gram_4], content


def calc_gram(single_picture_tensor_conv):
    wTimesH = int(single_picture_tensor_conv.get_shape()[0] * single_picture_tensor_conv.get_shape()[1])
    numFilters = int(single_picture_tensor_conv.get_shape()[2])
    tensor_conv_reshape = tf.reshape(single_picture_tensor_conv, (wTimesH, numFilters))
    return tf.matmul(tf.transpose(tensor_conv_reshape), tensor_conv_reshape)


def calc_content_loss(graph, content_input):
    tensor_conv = graph.get_tensor_by_name(VGG_CONTENT_LAYER)

    #amount_pictures = int(tensorshape_to_int_array(tensor_conv.get_shape())[0] / 2.0)
    amount_pictures = tensorshape_to_int_array(tensor_conv.get_shape())[0]

    content_l = 0.0
    for i in range(amount_pictures) :
        #content_l += tf.reduce_sum(tf.square(tensor_conv[i] - tensor_conv[i + amount_pictures]), name='content_loss')
        content_l += tf.reduce_sum(tf.square(tensor_conv[i] - content_input[i]), name='content_loss')

    return content_l


def calc_style_loss_64(graph, precomputed_style_grams):
    tensor_conv1_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_1)
    tensor_conv2_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_2)
    tensor_conv3_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_3)
    tensor_conv4_1 = graph.get_tensor_by_name(VGG_STYLE_TENSOR_4)
    #tensor_conv5_1 = graph.get_tensor_by_name("import/conv5_2/Relu:0")

    #amount_pictures = int(tensorshape_to_int_array(tensor_conv1_1.get_shape())[0] / 2.0)
    amount_pictures = tensorshape_to_int_array(tensor_conv1_1.get_shape())[0]

    style_l = 0.0
    for i in range(amount_pictures):

        tensor_gen_gram1_1 = calc_gram(tensor_conv1_1[i])
        tensor_gen_gram2_1 = calc_gram(tensor_conv2_1[i])
        tensor_gen_gram3_1 = calc_gram(tensor_conv3_1[i])
        tensor_gen_gram4_1 = calc_gram(tensor_conv4_1[i])
        #tensor_gen_gram5_1 = calc_gram(tensor_conv5_1[0])

        tensor_style_gram1_1 = precomputed_style_grams[0]
        tensor_style_gram2_1 = precomputed_style_grams[1]
        tensor_style_gram3_1 = precomputed_style_grams[2]
        tensor_style_gram4_1 = precomputed_style_grams[3]
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

        style_l += style_loss1_1 + style_loss2_1 + style_loss3_1 + style_loss4_1
    return style_l


def main():

    input_images, content_input_images = load_pictures_for_feed("\\batch", recursive=True)
    style_red, avg_style_red = load_image("\\styles\\style.jpg", between_01=True, substract_mean=False)

    pre_style_grams, pre_content_tensor = precompute_style_gram(style_red, content_input_images)

    gen_graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars = build_gen_graph_deep(input_pictures=BATCH_SIZE)
    gen_image = gen_graph['output']

    #style_image = tf.placeholder('float32', [1, 224, 224,3], name="style_image")
    content_layer = tf.placeholder('float32', [BATCH_SIZE, 224, 224,64], name="content_layer")

    batch = tf.slice(gen_image, [0, 4, 4, 0], [-1, 224, 224, -1])
    #batch = tf.concat(0, [batch, content_input])

    graph = load_vgg_input(batch)

    content_loss = 7.5 * calc_content_loss(graph, content_layer)
    style_loss = 1e2 * calc_style_loss_64(graph, pre_style_grams)
    loss = content_loss + style_loss

    learning_rate = 0.001
    var_learning_rate = tf.placeholder("float32")

    image_counter = 0
    assert len(input_images) >= BATCH_SIZE

    feed = {}
    feed[input_image] = input_images[image_counter : image_counter + BATCH_SIZE]
    #feed[content_input] = content_input_images[image_counter : image_counter + BATCH_SIZE]
    feed[content_layer] = pre_content_tensor[image_counter: image_counter + BATCH_SIZE]
    # feed[style_image] = style_red.reshape(1, 224, 224,3)
    feed[var_learning_rate] = learning_rate

    image_counter = (image_counter + 4) % len(input_images)
    if image_counter + 4 > len(input_images) :
        image_counter = 0

    with tf.Session() as sess:

        # set log directory
        #summary_writer = tf.train.SummaryWriter(project_path + log_train,graph_def=sess.graph_def)



        #optimizer = tf.train.MomentumOptimizer(learning_rate=var_learning_rate, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=var_learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        variables = variables_gen_filter + variables_gen_bias + variables_scalars
        train_step = optimizer.minimize(loss, var_list=variables)

        print('number of variables : ' + str(len(tf.trainable_variables())));

        init = tf.global_variables_initializer()
        sess.run(init, feed)


        loading_directory = "\\version_50_k"
        saving_directory = "\\version_50_k"
        starting_pic_num = 0

        saver = create_saver(sess)
        load_gen_last_checkpoint(sess, saver, path=loading_directory)


        i = 0
        last_l = sess.run(loss, feed_dict=feed)
        last_cl = sess.run(content_loss, feed_dict=feed)
        last_sl = sess.run(style_loss, feed_dict=feed)
        #last_bl = sess.run(black_loss, feed_dict=feed)
        #last_wl = sess.run(weight_loss, feed_dict=feed)

        start_training_time = time.time()
        last_training_checkpoint_time = start_training_time

        neg_loss_counter = 0
        avoid_save_loss = -1.0;

        restore= False
        last_saved_iteration = 0
        for i in range(0):
            if(i % 10 == 0):
                print(i)

            if i % 250 == 0:
                l = sess.run(loss, feed_dict=feed)

                if (last_l - l ) < 0 and i != 0:
                    avoid_save_loss = last_l
                    neg_loss_counter += 1
                    print('neg loss -> counter increase :' + str(neg_loss_counter))
                    if neg_loss_counter == 3 :
                        learning_rate /= 10.0
                        neg_loss_counter = 0
                        restore = True
                        print('neg loss -> reset counters to 0')
                        print("new learning rate : " + str(learning_rate))
                else:
                    if avoid_save_loss != -1.0 :
                        if l < avoid_save_loss:
                            avoid_save_loss = -1.0
                            neg_loss_counter = 0;
                            print("loss reached best result again")
                            print("reset counter to 0")
                        else:
                            print("avoid saving until loss becomes smaller again:" + str(l - avoid_save_loss))

                print('learning rate : ' + str(learning_rate))

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

                #bl = sess.run(black_loss, feed_dict=feed)
                #print('black_loss : ' + str(bl))
                #print('black_loss_improvement : ' + str((last_bl - bl) / last_bl))
                #last_bl = bl
                #wl = sess.run(weight_loss, feed_dict=feed)
                #print('weight_loss : ' + str(wl))
                #print('weight_loss_improvement : ' + str((last_wl - wl) / last_wl))
                #last_wl=wl

                t = time.time()
                print('training time: ' + time_to_str(t - start_training_time))
                print('training time since last checkpoint: ' + time_to_str(t - last_training_checkpoint_time))
                last_training_checkpoint_time = t

                save_image(saving_directory, '\\im' + str(i + starting_pic_num), sess.run(gen_image, feed_dict=feed), to255=True)

                if restore == False:
                    if avoid_save_loss == -1 :
                        save_gen_checkpoint(sess, saver, path=saving_directory)
                        last_saved_iteration = i
                else:
                    print("Restoring last checkpoint -> iteration : " + str(last_saved_iteration))
                    load_gen_last_checkpoint(sess, saver, path=saving_directory)
                    restore = False

            sess.run(train_step, feed_dict=feed)

            feed[input_image] = input_images[image_counter : image_counter + BATCH_SIZE]
            feed[content_layer] = pre_content_tensor[image_counter: image_counter + BATCH_SIZE]

            image_counter = (image_counter + 4) % len(input_images)
            if image_counter + 4 > len(input_images):
                image_counter = 0

        save_image(saving_directory, '\\im' + str(i + starting_pic_num + 1), sess.run(gen_image, feed_dict=feed), to255=True)
        print(sess.run(loss, feed_dict=feed))
        if avoid_save_loss == -1:
            save_gen_checkpoint(sess, saver, path=saving_directory)
            export_gen_graph(sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory)
        else:
            print("Restoring last checkpoint -> iteration : " + str(last_saved_iteration))
            load_gen_last_checkpoint(sess, saver, path=saving_directory)
            print("export pb-File")
            export_gen_graph(sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory)


def export_checkpoint_to_android():

    gen_graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars = build_gen_graph_deep(
        input_pictures=BATCH_SIZE)

    loading_directory = "\\version_50_k"
    saving_directory = "\\version_50_k"

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = create_saver(sess)
        load_gen_last_checkpoint(sess, saver, path=loading_directory)

        export_gen_graph(sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory)


def test_android_gen():
    full_path = output_generator + '\\checkStyleContent_20_plus_43_k'

    cat_gen, avg_cat_gen = load_image("\\cat.jpg", between_01=True, substract_mean=False, output_size=304)

    print('load generator')
    with open(project_path + full_path + '\\gen_export.pb', mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    input_image = tf.placeholder("float32", (1, 304, 304, 3), "input")
    tf.import_graph_def(graph_def, input_map={"ph_input_image": input_image})
    # print("graph loaded from disk")
    print('Done')

    output = tf.get_default_graph().get_tensor_by_name('import/output:0')

    feed = {}
    feed[input_image] = cat_gen.reshape(1, 304, 304, 3)

    with tf.Session() as sess :
        init = tf.global_variables_initializer()
        sess.run(init)
        x = sess.run(output, feed_dict=feed)
        save_image('\\generator_load', '\\test', x, True)


#main()
#transform(np.reshape(elch, [1,224,224,3]), '\\tmp\\checkStyleContent_10_plus_12', '\\checkpoint.data.meta', '\\style_10_plus_12', '\\elch.jpg')
export_checkpoint_to_android()
#test_android_gen()