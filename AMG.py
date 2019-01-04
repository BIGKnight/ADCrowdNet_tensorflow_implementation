import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
model_path = '/home/zzn/PycharmProjects/ADCrowdNet/vgg_pre-trained_variable_list/vgg_16.ckpt'


def AMG_back_end_arg_scope(weight_decay=4e-4, std=3, batch_norm_var_collection="moving_vars"):
    instance_norm_params = {
        # "decay": 0.9997,
        "epsilon": 1e-6,
        "activation_fn": tf.nn.relu,
        "trainable": True,
        "variables_collections": {
            "beta": None,
            "gamma": None,
            "moving_mean": [batch_norm_var_collection],
            "moving_variance": [batch_norm_var_collection]},
        "outputs_collections": {

        }
    }
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.glorot_uniform_initializer(),
                        activation_fn=tf.nn.relu) as sc:
        return sc


def global_average_pooling(features, name):
    return tf.reduce_mean(features, reduction_indices=[1, 2], name=name)


def AMG_front_end(features):
    logits, end_points = nets.vgg.vgg_16(features)
    assert (os.path.isfile(model_path))
    variables_to_restore = slim.contrib_framework.get_variables_to_restore(
        include=['vgg_16/conv1',
                 'vgg_16/pool1',
                 'vgg_16/conv2',
                 'vgg_16/pool2',
                 'vgg_16/conv3',
                 'vgg_16/pool3',
                 'vgg_16/conv4'])
    init_fn = slim.contrib_framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
    front_end = end_points['vgg_16/conv4/conv4_3']
    return front_end, init_fn


def inception_structure(features, index, output_nums):
    with tf.variable_scope(name_or_scope='inception_' + str(index)):
        part_1 = slim.conv2d(features, output_nums, [3, 3], rate=1, scope='part_1')
        part_2 = slim.conv2d(features, output_nums, [3, 3], rate=3, scope='part_2')
        part_3 = slim.conv2d(features, output_nums, [3, 3], rate=6, scope='part_3')
        part_4 = slim.conv2d(features, output_nums, [3, 3], rate=9, scope='part_4')
    concat = tf.concat([part_1, part_2, part_3, part_4], axis=3, name="concat_" + str(index))
    return concat


def AMG_back_end(featuers):
    with slim.arg_scope(AMG_back_end_arg_scope()):
        net = inception_structure(featuers, 1, 256)
        net = slim.conv2d(net, 256, [1, 1], rate=1, scope='phase_1')
        net = inception_structure(net, 2, 128)
        net = slim.conv2d(net, 128, [1, 1], rate=1, scope='phase_2')
        net = inception_structure(net, 3, 64)
        net = slim.conv2d(net, 2, [1, 1], rate=1, scope='phase_3')
    return net


def AMG(features, batch_size):
    # the front_end of the AMG phase
    # we assume the inputs features were finely pre-processed
    front_end, init_vgg16_fn = AMG_front_end(features)

    # the back_end of the AMG phase
    back_end = AMG_back_end(front_end)
    # generate_AMG_output
    with tf.variable_scope(name_or_scope="generate_AMG_output"):
        wb_wc = global_average_pooling(back_end, name="global_average_pooling")
        pb_pc = tf.nn.softmax(wb_wc, 1, name="soft_max")
        batch = tf.split(back_end, batch_size, 0)
        p_per_image = tf.split(pb_pc, batch_size, 0)
        batch_list = []
        for i, j in zip(p_per_image, batch):
            i = tf.squeeze(i)  # because the split op will retain the shape of the inputs
            fb, fc = tf.split(j, 2, 3)
            batch_list.append(i[0] * fb + i[1] * fc)
        AMG_output = tf.concat(batch_list, 0)
    # the author said the output of the AMG need to be normalized
    # but likely not indicated the specific methods of normalization,
    # i adopted sigmoid function to normalize the whole output
    AMG_output = tf.sigmoid(AMG_output)
    return AMG_output, init_vgg16_fn