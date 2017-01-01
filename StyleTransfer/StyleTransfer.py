import scipy.io
import scipy.misc
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import numpy as np
import sys


from matplotlib.pyplot import imshow

synset = [l.strip() for l in open(
    'C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\model\\synset.txt').readlines()]





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

def save_image(path, image, to255=False):
    # Output should add back the mean.
    #image = image + MEAN_VALUES
    # Get rid of the first useless dimension, what remains is the image.
    image = image[0]
    if to255 == True :
        image = image * 255.0
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave(path, image)

cat = load_image('C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\images\\cat.jpg')
elch = load_image('C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\images\\elch.jpg')
style = load_image('C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\images\\style.jpg')

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


def load_vgg_input( images, path='C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\model\\vgg.tfmodel'):
    with open(path, mode='rb') as f:
        fileContent = f.read()
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)
    #images = tf.placeholder("float32", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images": images})
    #print("graph loaded from disk")
    return tf.get_default_graph(), images


def _relu(conv2d_layer):
    return tf.nn.relu(conv2d_layer)

variables_gen_filter = []
variables_gen_bias = []

def _conv2d(prev_layer):
    #W = tf.Variable(2*np.random.rand(3,3,3,3)-1, dtype="float32", name="W")
    b = tf.Variable(np.random.rand(3), dtype="float32", name="b")
    #variables_gen_filter.append(W)
    variables_gen_bias.append(b)
    #return tf.nn.conv2d(prev_layer, filter=W, strides=[1,1,1,1], padding='SAME')+b
    return prev_layer+b

def _conv2d_relu(prev_layer):
    return _relu(_conv2d(prev_layer))

def build_gen_graph():
    graph = {}
    #input_image = tf.constant(cat.reshape(1,224,224,3), dtype="float32")
    #input_image = tf.Variable(cat.reshape(1,224,224,3), trainable=False, dtype="float32", name="input_image")
    #graph['gen_input'] = tf.Variable(input_image, trainable=False)
    input_image = tf.placeholder('float32', [1, 224,224,3], name="input_image")
    #graph['conv1_1'] = _conv2d_relu(input_image)
    #graph['conv2_1'] = _conv2d_relu(graph['conv1_1'])
    #graph['conv3_1'] = _conv2d_relu(graph['conv2_1'])
    #graph['conv4_1'] = _conv2d_relu(graph['conv3_1'])
    #graph['conv5_1'] = _conv2d_relu(graph['conv4_1'])
    graph['output'] = _conv2d_relu(input_image)
    return graph, input_image


def calc_content_loss(graph, layer = "import/conv4_2/Relu:0"):
    tensor_conv = graph.get_tensor_by_name(layer)
    content_loss = tf.reduce_sum(tf.square(tensor_conv[0] - tensor_conv[1]), name='content_loss')
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

gen_graph, input_image = build_gen_graph()
gen_image = gen_graph['output']

style_image = tf.placeholder('float32', [1,224,224,3], name="style_image")
#style_image = tf.Variable(style.reshape(1,224,224,3), trainable=False, dtype="float32", name="style_image")

#batch = tf.nn.sigmoid(gen_image)
batch = gen_image
batch = tf.concat(0, [batch, tf.div(input_image, 255.0)])
batch = tf.concat(0, [batch, tf.div(style_image, 255.0)])
assert batch.get_shape() == (3, 224, 224, 3)

graph, images = load_vgg_input(batch)

content_loss = calc_content_loss(graph)
style_loss = calc_style_loss_64(graph)
loss = tf.cast(content_loss, tf.float64)

feed = {}
feed[input_image] = cat.reshape(1,224,224,3)
feed[style_image] = style.reshape(1,224,224,3)

with tf.Session() as sess:

    # set log directory
    summary_writer = tf.train.SummaryWriter(
        'C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\logs',
        graph_def=sess.graph_def)

    # 0 generiert
    # 1 content
    # 2 style

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    variables = variables_gen_filter + variables_gen_bias
    train_step = optimizer.minimize(loss, var_list=variables)

    print(len(tf.trainable_variables()));

    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(20):
        if i % 1 == 0:
            print(sess.run(loss, feed_dict=feed))
            #print(sess.run(variables_gen_filter[0]))
            print(sess.run(variables_gen_bias[0], feed_dict=feed))
            #print(sess.run(variables[0]))
            #save_image('C:\\Users\\ken\\uni\\05_UNI_WS_16-17\\Visual_Data\\DLVD_Project\\StyleTransfer\\output_images\\im' + str(i) + '.jpg', sess.run(gen_image))
            #print(sess.run(gen_graph['conv1_1'], feed_dict=feed))
        sess.run(train_step, feed_dict=feed)


