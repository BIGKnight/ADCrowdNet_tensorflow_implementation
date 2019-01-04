import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
model_path = '/home/zzn/PycharmProjects/ADCrowdNet/vgg_pre-trained_variable_list/vgg_16.ckpt'


def DME_front_end(features):
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


def deformable_conv2d(features, kernel_size, output_nums, stride, batch_size, image_height, image_weight):
    kernel_arg_nums = kernel_size[0] * kernel_size[1]
    args = slim.conv2d(features, 3 * kernel_arg_nums, kernel_size, stride, padding='SAME')
    offset = tf.slice(args, [0, 0, 0, 0], [batch_size, image_height, image_weight, 2 * kernel_arg_nums])
    mask = tf.slice(args, [0, 0, 0, 2 * kernel_arg_nums], [batch_size, image_height, image_weight, kernel_arg_nums])
    tf.nn.de

def DME_back_end(features):



def DME(features, batch_size, image_height, image_weight):
    front_end, init_DME_fn = DME_front_end(features)
