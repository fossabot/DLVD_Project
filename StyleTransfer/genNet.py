import numpy as np
import utils as utils
import config as conf

def _relu(tf, conv2d_layer):
    return tf.nn.relu(conv2d_layer)


def _conv2d(tf, variables_gen_filter, variables_gen_bias, prev_layer, i_num_channel = 3, o_num_filter = 3, strides=[1, 1, 1, 1], filter_size=3, pad='SAME', is_trainable = True):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    if is_trainable:
        W = tf.Variable(tf.truncated_normal([filter_size, filter_size ,i_num_channel, o_num_filter], dtype='float32', stddev=conf.INIT_STD_DEV, seed=conf.TRUNCATED_SEED), dtype="float32", name="W", trainable=is_trainable)
        #b = tf.Variable(tf.truncated_normal([o_num_filter], stddev=conf.INIT_STD_DEV, seed=conf.TRUNCATED_SEED, dtype='float32'), dtype="float32", name="b", trainable=is_trainable)
        variables_gen_filter.append(W)
        #variables_gen_bias.append(b)
    else:
        W = tf.constant(variables_gen_filter.pop(0), dtype="float32", name="W")
        #b = tf.constant(variables_gen_bias.pop(0), dtype="float32", name="b")
    #return tf.add(tf.nn.conv2d(prev_layer, filter=W, strides=strides, padding=pad), b)
    return tf.nn.conv2d(prev_layer, filter=W, strides=strides, padding=pad)


def _fract_conv2d(tf, variables_gen_filter, variables_gen_bias, prev_layer, strides, i_num_channel = 3, o_num_filter = 3, pad='SAME', filter_size=3, is_trainable = True):
    var = np.sqrt(2.0 / (filter_size * filter_size * i_num_channel))
    if is_trainable:
        W = tf.Variable(tf.truncated_normal([filter_size,filter_size,o_num_filter, i_num_channel], dtype='float32', stddev=conf.INIT_STD_DEV, seed=conf.TRUNCATED_SEED), dtype="float32", name="W", trainable=is_trainable)
        #b = tf.Variable(tf.truncated_normal([o_num_filter], dtype='float32', stddev=conf.INIT_STD_DEV, seed=conf.TRUNCATED_SEED), dtype="float32", name="b", trainable=is_trainable)
        variables_gen_filter.append(W)
        #variables_gen_bias.append(b)
    else:
        W = tf.constant(variables_gen_filter.pop(0), dtype="float32", name="W")
        #b = tf.constant(variables_gen_bias.pop(0), dtype="float32", name="b")
    shape = utils.tensorshape_to_int_array(prev_layer.get_shape())
    #return tf.add(tf.nn.conv2d_transpose(prev_layer, W, [shape[0], strides[1]*shape[1], strides[2]*shape[2], o_num_filter ] , strides, padding=pad), b)
    return tf.nn.conv2d_transpose(prev_layer, W, [shape[0], strides[1]*shape[1], strides[2]*shape[2], o_num_filter ] , strides, padding=pad)

def _instance_norm(tf, variable_scalars, x, epsilon=1e-3, is_trainable=True):
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


def _clip_2x2_border(tf, x):
    shape = utils.tensorshape_to_int_array(x.get_shape())
    height_end = shape[1] - 4
    with_end = shape[2] - 4
    tmp = tf.slice(x, [0, 2, 2, 0], [shape[0], height_end, with_end, shape[3]])
    return tmp


def build_gen_graph_deep(tf, trainable = True, variables_gen_filter = [], variables_gen_bias = [], variables_scalars = [], input_pictures = 1, input_resolution = 224):

    if trainable :
        variables_gen_filter = []
        variables_gen_bias = []
        variables_scalars = []

    graph = {}
    input_image = tf.placeholder('float32', [input_pictures, input_resolution, input_resolution, 3], name="ph_input_image")

    graph['conv1_0'] = _relu(tf,
                        _instance_norm(tf,
                            variables_scalars,
                            _conv2d(tf,
                                variables_gen_filter,
                                variables_gen_bias,
                                input_image,
                                filter_size=9,
                                o_num_filter=32,
                                is_trainable = trainable
                            ),
                            is_trainable=trainable
                        )
    )
    print(graph['conv1_0'].get_shape())

    graph['conv2_0'] = _relu(tf,
                        _instance_norm(tf,
                            variables_scalars,
                            _conv2d(tf,
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv1_0'],
                                strides=[1, conf.DOWN_SAMPLING, conf.DOWN_SAMPLING, 1],
                                i_num_channel = 32,
                                o_num_filter = 64,
                                is_trainable = trainable
                            ),
                        is_trainable=trainable
                        )
    )
    print(graph['conv2_0'].get_shape())

    graph['conv2_1'] = _relu(tf,
                        _instance_norm(tf,
                            variables_scalars,
                            _conv2d(tf,
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv2_0'],
                                strides=[1, conf.DOWN_SAMPLING, conf.DOWN_SAMPLING, 1],
                                i_num_channel=64,
                                o_num_filter=128,
                                is_trainable = trainable
                            ),
                        is_trainable=trainable
                        )
    )
    print(graph['conv2_1'].get_shape())

    graph['conv3_0_0'] = _relu(tf,
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
                                        variables_gen_filter,
                                        variables_gen_bias,
                                        graph['conv2_1'],
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
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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

    graph['conv3_1_0'] = _relu(tf,
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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

    graph['conv3_2_0'] = _relu(tf,
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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

    graph['conv3_2_1'] = tf.add(
                                graph['conv3_1_1'],
                                _instance_norm(tf,
                                    variables_scalars,
                                    _conv2d(tf,
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

    graph['conv3_3_0'] = _relu(tf,
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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

    graph['conv3_3_1'] = tf.add(
                                graph['conv3_2_1'],
                                _instance_norm(tf,
                                    variables_scalars,
                                    _conv2d(tf,
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

    graph['conv3_4_0'] = _relu(tf,
                            _instance_norm(tf,
                                variables_scalars,
                                _conv2d(tf,
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

    graph['conv3_4_1'] = tf.add(
                                graph['conv3_3_1'],
                                _instance_norm(tf,
                                    variables_scalars,
                                    _conv2d(tf,
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

    graph['conv4_0'] = _relu(tf,
                        _instance_norm(tf,
                            variables_scalars,
                            _fract_conv2d(tf,
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv3_4_1'],
                                [1, conf.DOWN_SAMPLING, conf.DOWN_SAMPLING, 1],
                                i_num_channel=128,
                                o_num_filter=64,
                                is_trainable = trainable
                            ),
                            is_trainable=trainable
                        )
    )
    print(graph['conv4_0'].get_shape())

    graph['conv4_1'] = _relu(tf,
                        _instance_norm(tf,
                            variables_scalars,
                            _fract_conv2d(tf,
                                variables_gen_filter,
                                variables_gen_bias,
                                graph['conv4_0'],
                                [1, conf.DOWN_SAMPLING, conf.DOWN_SAMPLING, 1],
                                i_num_channel=64,
                                o_num_filter=32,
                                is_trainable = trainable
                            ),
                            is_trainable=trainable
                        )
    )
    print(graph['conv4_1'].get_shape())

    graph['conv5_0'] = _instance_norm(tf,
                        variables_scalars,
                        _conv2d(tf,
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

    print(graph['conv5_0'].get_shape())

    graph['output'] = tf.add(tf.nn.tanh(graph['conv5_0']) * 150, 225./2, name='output')

    return graph, input_image, variables_gen_filter, variables_gen_bias, variables_scalars
