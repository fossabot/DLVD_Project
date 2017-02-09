import tensorflow as tf
import time
import random

import config as conf
import genNet as gn
import utils as utils
import vgg as vn

import networkIO as nio
import androidInterface as ai

import numpy as np

def precompute_style_gram(style_image, content_images):
    print("Precompute style tensors")
    graph = tf.Graph()
    with graph.as_default() as g:
        inp = tf.placeholder("float32", [None, conf.VGG_INPUT_RESOLUTION, conf.VGG_INPUT_RESOLUTION, 3])
        vn.load_vgg_input(tf, inp)

        tensor_conv1_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_1)
        tensor_conv2_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_2)
        tensor_conv3_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_3)
        tensor_conv4_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_4)

        tensor_style_gram1_1 = calc_gram(tensor_conv1_1[0])
        tensor_style_gram2_1 = calc_gram(tensor_conv2_1[0])
        tensor_style_gram3_1 = calc_gram(tensor_conv3_1[0])
        tensor_style_gram4_1 = calc_gram(tensor_conv4_1[0])

        feed = {}
        feed[inp] = style_image.reshape(1, conf.VGG_INPUT_RESOLUTION, conf.VGG_INPUT_RESOLUTION, 3)
        with tf.Session() as sess :
            gram_1 = sess.run(tensor_style_gram1_1, feed_dict=feed)
            gram_2 = sess.run(tensor_style_gram2_1, feed_dict=feed)
            gram_3 = sess.run(tensor_style_gram3_1, feed_dict=feed)
            gram_4 = sess.run(tensor_style_gram4_1, feed_dict=feed)

        tensor_conv = graph.get_tensor_by_name(conf.VGG_CONTENT_LAYER)

        feed = {}
        content = []
        counter = 0
        batch_size = conf.PRECOMPUTE_BATCH_SIZE
        brek_now = False
        while True :
            if counter % 10 == 0:
                print("Number of pictures already computed : " + str(counter))

            if counter + batch_size > len(content_images):
                batch_size = len(content_images) - counter
                brek_now = True
                if batch_size == 0 :
                    break;

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
    tensor_conv = graph.get_tensor_by_name(conf.VGG_CONTENT_LAYER)

    #amount_pictures = int(tensorshape_to_int_array(tensor_conv.get_shape())[0] / 2.0)
    amount_pictures = utils.tensorshape_to_int_array(tensor_conv.get_shape())[0]

    content_l = 0.0
    for i in range(amount_pictures) :
        #content_l += tf.reduce_sum(tf.square(tensor_conv[i] - tensor_conv[i + amount_pictures]), name='content_loss')
        content_l += tf.reduce_sum(tf.square(tensor_conv[i] - content_input[i]), name='content_loss')

    return content_l


def calc_style_loss_64(graph, precomputed_style_grams):
    tensor_conv1_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_1)
    tensor_conv2_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_2)
    tensor_conv3_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_3)
    tensor_conv4_1 = graph.get_tensor_by_name(conf.VGG_STYLE_TENSOR_4)
    #tensor_conv5_1 = graph.get_tensor_by_name("import/conv5_2/Relu:0")

    #amount_pictures = int(tensorshape_to_int_array(tensor_conv1_1.get_shape())[0] / 2.0)
    amount_pictures = utils.tensorshape_to_int_array(tensor_conv1_1.get_shape())[0]

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



        s = utils.tensorshape_to_int_array(tensor_conv1_1.get_shape())
        style_loss1_1_nominator = tf.reduce_sum(
            tf.pow(tensor_gen_gram1_1 - tensor_style_gram1_1, 2.0))
        style_loss1_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
        style_loss1_1 = tf.div(style_loss1_1_nominator, style_loss1_1_denominator)

        s = utils.tensorshape_to_int_array(tensor_conv2_1.get_shape())
        style_loss2_1_nominator = tf.reduce_sum(
            tf.pow(tensor_gen_gram2_1 - tensor_style_gram2_1, 2.0))
        style_loss2_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
        style_loss2_1 = tf.div(style_loss2_1_nominator, style_loss2_1_denominator)

        s = utils.tensorshape_to_int_array(tensor_conv3_1.get_shape())
        style_loss3_1_nominator = tf.reduce_sum(
            tf.pow(tensor_gen_gram3_1 - tensor_style_gram3_1, 2.0))
        style_loss3_1_denominator = 4.0 * ((s[1] * s[2]) ** 2) * (s[3] ** 2.0)
        style_loss3_1 = tf.div(style_loss3_1_nominator, style_loss3_1_denominator)

        s = utils.tensorshape_to_int_array(tensor_conv4_1.get_shape())
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


def calc_tv_loss(gen_image):
    y_tv = tf.nn.l2_loss(gen_image[:, 1:, :, :] - gen_image[:, :conf.INPUT_RESOLUTION - 1, :, :])
    x_tv = tf.nn.l2_loss(gen_image[:, :, 1:, :] - gen_image[:, :, :conf.INPUT_RESOLUTION - 1, :])
    return 2 * (x_tv + y_tv)

def main():

    input_images, content_input_images = utils.load_pictures_for_feed("\\batch", recursive=True, gen_res=conf.INPUT_RESOLUTION, content_res=conf.VGG_INPUT_RESOLUTION)

    print("Shuffle inputs")
    random.seed(conf.SEED)
    random.shuffle(input_images)
    random.seed(conf.SEED)
    random.shuffle(content_input_images)
    print("Done")

    style_red, avg_style_red = utils.load_image("\\styles\\rain_princess.jpg", between_01=True, substract_mean=False)

    pre_style_grams, pre_content_tensor = precompute_style_gram(style_red, content_input_images)

    gen_graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars = gn.build_gen_graph_deep(tf, input_pictures=conf.BATCH_SIZE, width_res=conf.INPUT_RESOLUTION)
    gen_image = gen_graph['output']

    pre_content_tensor_shape= np.shape(pre_content_tensor)
    content_layer = tf.placeholder('float32'
                                ,[conf.BATCH_SIZE, pre_content_tensor_shape[1], pre_content_tensor_shape[2], pre_content_tensor_shape[3]]
                                ,name="content_layer")

    #gen_shape = utils.tensorshape_to_int_array(gen_image.get_shape())
    #cut_1 = int((gen_shape[1] - conf.VGG_INPUT_RESOLUTION) / 2)
    #cut_2 = int((gen_shape[2] - conf.VGG_INPUT_RESOLUTION) / 2)
    #batch = tf.slice(gen_image, [0, cut_1, cut_2, 0], [gen_shape[0], conf.VGG_INPUT_RESOLUTION, conf.VGG_INPUT_RESOLUTION, gen_shape[3]])

    batch = gen_image / 255.0
    print(utils.tensorshape_to_int_array(batch.get_shape()))

    graph = vn.load_vgg_input(tf, batch)

    content_loss = conf.CONTENT_WEIGHT * calc_content_loss(graph, content_layer)
    style_loss = conf.STYLE_WEIGHT * calc_style_loss_64(graph, pre_style_grams)
    tv_loss = conf.TV_WEIGHT * calc_tv_loss(gen_image)
    loss = content_loss + style_loss + tv_loss

    learning_rate = conf.LEARNING_RATE
    var_learning_rate = tf.placeholder("float32")

    image_counter = 0
    assert len(input_images) >= conf.BATCH_SIZE

    feed = {}
    feed[input_image] = input_images[image_counter : image_counter + conf.BATCH_SIZE]
    #feed[content_input] = content_input_images[image_counter : image_counter + BATCH_SIZE]
    feed[content_layer] = pre_content_tensor[image_counter: image_counter + conf.BATCH_SIZE]
    # feed[style_image] = style_red.reshape(1, 224, 224,3)
    feed[var_learning_rate] = learning_rate

    image_counter = (image_counter + conf.BATCH_SIZE) % len(input_images)
    if image_counter + conf.BATCH_SIZE > len(input_images) :
        image_counter = 0

    with tf.Session() as sess:

        # set log directory
        #summary_writer = tf.train.SummaryWriter(conf.project_path + conf.log_train, graph_def=sess.graph_def)

        #optimizer = tf.train.MomentumOptimizer(learning_rate=var_learning_rate, momentum=0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=var_learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        variables = variables_gen_filter + variables_gen_bias + variables_scalars
        train_step = optimizer.minimize(loss, var_list=variables)

        print('number of variables : ' + str(len(tf.trainable_variables())));

        init = tf.global_variables_initializer()
        sess.run(init, feed)


        loading_directory = "\\version_60_k"
        saving_directory = "\\version_60_k"
        starting_pic_num = 20500

        saver = nio.create_saver(tf, sess)
        nio.load_gen_last_checkpoint(tf, sess, saver, path=loading_directory)


        i = 0
        last_l = sess.run(loss, feed_dict=feed)
        last_cl = sess.run(content_loss, feed_dict=feed)
        last_sl = sess.run(style_loss, feed_dict=feed)
        last_tvl = sess.run(tv_loss, feed_dict=feed)
        #last_wl = sess.run(weight_loss, feed_dict=feed)

        start_training_time = time.time()
        last_training_checkpoint_time = start_training_time

        neg_loss_counter = 0
        avoid_save_loss = -1.0;

        restore= False
        last_saved_iteration = 0
        for i in range(19500):
            if(i % 10 == 0):
                print(i)

            if i % 250 == 0:
                l = sess.run(loss, feed_dict=feed)

                if (last_l - l ) < 0 and i != 0:
                    avoid_save_loss = last_l
                    neg_loss_counter += 1
                    print('neg loss -> counter increase :' + str(neg_loss_counter))
                    if neg_loss_counter == 5 :
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

                tvl = sess.run(tv_loss, feed_dict=feed)
                print('tv_loss : ' + str(tvl))
                print('tv_loss_improvement : ' + str((last_tvl - tvl) / last_tvl))
                last_tvl = tvl

                t = time.time()
                print('training time: ' + utils.time_to_str(t - start_training_time))
                print('training time since last checkpoint: ' + utils.time_to_str(t - last_training_checkpoint_time))
                last_training_checkpoint_time = t

                utils.save_image(saving_directory, '\\im' + str(i + starting_pic_num), sess.run(gen_image, feed_dict=feed), to255=False)

                if restore == False:
                    if avoid_save_loss == -1 :
                        nio.save_gen_checkpoint(sess, saver, path=saving_directory)
                        last_saved_iteration = i
                else:
                    print("Restoring last checkpoint -> iteration : " + str(last_saved_iteration))
                    nio.load_gen_last_checkpoint(tf, sess, saver, path=saving_directory)
                    restore = False

            sess.run(train_step, feed_dict=feed)

            feed[input_image] = input_images[image_counter : image_counter + conf.BATCH_SIZE]
            feed[content_layer] = pre_content_tensor[image_counter: image_counter + conf.BATCH_SIZE]

            image_counter = (image_counter + conf.BATCH_SIZE) % len(input_images)
            if image_counter + conf.BATCH_SIZE > len(input_images):
                image_counter = 0

        utils.save_image(saving_directory, '\\im' + str(i + starting_pic_num + 1), sess.run(gen_image, feed_dict=feed), to255=False)
        print(sess.run(loss, feed_dict=feed))
        if avoid_save_loss == -1:
            nio.save_gen_checkpoint(sess, saver, path=saving_directory)
            ai.export_gen_graph(tf, sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory)
        else:
            print("Restoring last checkpoint -> iteration : " + str(last_saved_iteration))
            nio.load_gen_last_checkpoint(tf, sess, saver, path=saving_directory)
            print("export pb-File")
            ai.export_gen_graph(tf, sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory)

#main()
#ai.export_checkpoint_to_android(tf)
ai.test_android_gen(tf)