import config as conf
import genNet as gn
import utils as utils
import networkIO as nio


def export_gen_graph(tf, sess, variables_filter, variables_bias, variables_scalars, path, name="gen_export.pb", width=224, ratio=1.0) :

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
        gn.build_gen_graph_deep(tf, trainable=False, variables_gen_filter=var_gen_filter_new, variables_gen_bias=var_gen_bias_new, variables_scalars=var_gen_scalars_new, width_res=width, ratio=ratio)

        #saver = tf.train.Saver(tf.all_variables())
        utils.make_sure_path_exists(conf.project_path + conf.output_generator + path)
        with tf.Session() as new_sess:
            init = tf.global_variables_initializer()
            new_sess.run(init)
            #summary_writer = tf.train.SummaryWriter(project_path + log_generator, graph_def=new_sess.graph_def)

            #saver.save(new_sess, project_path + "\\android_exports" + path + name)
            tf.train.write_graph(tf.get_default_graph(), conf.project_path + conf.output_generator + path, name, as_text=False)


def export_checkpoint_to_android(tf):

    gen_graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars = gn.build_gen_graph_deep(tf,
                                            input_pictures=conf.BATCH_SIZE, width_res=conf.INPUT_RESOLUTION)

    loading_directory = "\\version_59_k"
    saving_directory = "\\version_59_k"

    width = conf.ANDROID_EMULATOR_WIDTH
    ratio = conf.ANDROID_EMULATOR_RATIO
    height = int(width * ratio)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = nio.create_saver(tf, sess)
        nio.load_gen_last_checkpoint(tf, sess, saver, path=loading_directory)

        export_gen_graph(tf, sess, variables_gen_filter, variables_gen_bias, variables_scalars, saving_directory,
                         name='gen_export_' + str(width) + '_' + str(height) +'.pb', width=width, ratio=ratio)


def test_android_gen(tf):
    load_and_save_path = '\\version_59_k'
    full_path = conf.output_generator + load_and_save_path

    image = "\\skyline_high_res.jpg"
    width = conf.ANDROID_EMULATOR_WIDTH
    ratio = conf.ANDROID_EMULATOR_RATIO
    height = int(width * ratio)

    content, avg_content_gen = utils.load_image(image, between_01=True, substract_mean=False, width=width, ratio=ratio)

    print('load generator')
    with open(conf.project_path + full_path + '\\gen_export_' + str(width) + '_' + str(height) +'.pb', mode='rb') as f:
        fileContent = f.read()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fileContent)

    input_image = tf.placeholder("float32", (1, height, width, 3), "input")
    tf.import_graph_def(graph_def, input_map={"ph_input_image": input_image})
    # print("graph loaded from disk")
    print('Done')

    output = tf.get_default_graph().get_tensor_by_name('import/output:0')

    feed = {}
    feed[input_image] = content.reshape(1, height, width, 3)

    with tf.Session() as sess :
        init = tf.global_variables_initializer()
        sess.run(init)
        x = sess.run(output, feed_dict=feed)
        utils.save_image(load_and_save_path, '\\test', x)
