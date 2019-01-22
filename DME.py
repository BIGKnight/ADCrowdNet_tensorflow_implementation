import tensorflow as tf
import os
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
model_path = '/home/zzn/PycharmProjects/ADCrowdNet/vgg_pre-trained_variable_list/vgg_16.ckpt'


# def DME_front_end(features):
#     logits, end_points = nets.vgg.vgg_16(features)
#     assert (os.path.isfile(model_path))
#     variables_to_restore = slim.contrib_framework.get_variables_to_restore(
#         include=['vgg_16/conv1',
#                  'vgg_16/pool1',
#                  'vgg_16/conv2',
#                  'vgg_16/pool2',
#                  'vgg_16/conv3',
#                  'vgg_16/pool3',
#                  'vgg_16/conv4'
#                  ])
#     init_fn = slim.contrib_framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
#     front_end = end_points['vgg_16/conv4/conv4_3']
#     return front_end, init_fn


def DME_inception(features, index,  output_nums):
    with tf.variable_scope(name_or_scope='inception_' + str(index)):
        part_1 = slim.conv2d(features, output_nums, 3, 1, data_format='NCHW', scope='part_1')
        part_2 = slim.conv2d(features, output_nums, 5, 1, data_format='NCHW', scope='part_2')
        part_3 = slim.conv2d(features, output_nums, 7, 1, data_format='NCHW', scope='part_3')
    output = tf.concat([part_1, part_2, part_3], axis=1, name='concat_' + str(index))
    return output


def DME_back_end(features):
    features_transpose = tf.transpose(features, [0, 3, 1, 2])
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.leaky_relu):
        net_inception_1 = DME_inception(features_transpose, 1, 256)
        net_conv_1x1_1 = slim.conv2d(net_inception_1, 256, 1, 1, data_format='NCHW', scope='conv1x1_1')
        net_inception_2 = DME_inception(net_conv_1x1_1, 2, 128)
        net_conv_1x1_2 = slim.conv2d(net_inception_2, 128, 1, 1, data_format='NCHW', scope='conv1x1_2')
        net_inception_3 = DME_inception(net_conv_1x1_2, 3, 64)
        net_conv_1x1_3 = slim.conv2d(net_inception_3, 1, 1, 1, data_format='NCHW', scope='conv1x1_3')
    output = tf.transpose(net_conv_1x1_3, [0, 2, 3, 1])
    return output


def DME_model(features):
    features = slim.instance_norm(features, activation_fn=tf.nn.relu)
    # front_end, init_vgg16_fn = DME_front_end(features)
    _, end_points = nets.vgg.vgg_16(features)
    front_end = end_points['vgg_16/conv4/conv4_3']
    feature_map = DME_back_end(front_end)
    # return output, init_vgg16_fn

    return feature_map