
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
# from deformable_conv2d_op.deformable_conv2d import deformable_conv2d_op
import os.path as osp
from tensorflow.python.framework import ops
import math

model_path = '/home/zzn/PycharmProjects/ADCrowdNet/vgg_pre-trained_variable_list/vgg_16.ckpt'

# filename = osp.join(osp.dirname(__file__), 'deformable_conv2d.so')
deformable_conv2d_module = tf.load_op_library('./deformable_conv2d_op/deformable_conv2d.so')
deformable_conv2d_op = deformable_conv2d_module.deformable_conv2d
deformable_conv2d_grad_op = deformable_conv2d_module.deformable_conv2d_back_prop


@ops.RegisterGradient("DeformableConv2D")
def _deformable_conv2d_back_prop(op, grad):
    """The gradients for `deform_conv`.
        Args:
        op: The `deform_conv` `Operation` that we are differentiating, which we can use
        to find the inputs and outputs of the original op.
        grad: Gradient with respect to the output of the `roi_pool` op.
        Returns:
        Gradients with respect to the input of `deform_conv`.
    """
    data = op.inputs[0]
    filter = op.inputs[1]
    offset = op.inputs[2]
    mask = op.inputs[3]
    '''
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: {'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    strides = op.get_attr('strides')
    dilations = op.get_attr('dilations')
    data_format = op.get_attr('data_format')
    im2col_step = op.get_attr('im2col_step')
    no_bias = op.get_attr('no_bias')
    pads = op.get_attr('padding')
    num_groups = op.get_attr('num_groups')
    deformable_groups = op.get_attr('deformable_groups')
    '''
    REGISTER_OP("DeformableConv2DBackProp")
        .Input("input: T")
        .Input("filter: T")
        .Input("offset: T")
        .Input("mask: T")
        .Input("out_grad: T")
        .Output("x_grad: T")
        .Output("filter_grad: T")
        .Output("offset_grad: T")
        .Output("mask_grad: T")
        .Attr("T: {float, double}")
        .Attr("strides: list(int)")
        .Attr("num_groups: int")
        .Attr("deformable_groups: int")
        .Attr("im2col_step: int")
        .Attr("no_bias: bool = true")
        .Attr(GetPaddingAttrString())
        .Attr("data_format: { 'NCHW' } = 'NCHW' ")
        .Attr("dilations: list(int) = [1, 1, 1, 1]")
    '''
    # compute gradient
    data_grad, filter_grad, offset_grad, mask_grad = deformable_conv2d_grad_op(
        input=data,
        filter=filter,
        offset=offset,
        mask=mask,
        out_grad=grad,
        strides=strides,
        num_groups=num_groups, deformable_groups=deformable_groups, im2col_step=im2col_step,
        no_bias=no_bias,
        padding=pads,
        data_format=data_format,
        dilations=dilations)
    return [data_grad, filter_grad, offset_grad, mask_grad] # List of 4 Tensor, since we have 4 input


#
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
#                  'vgg_16/conv4'])
#     init_fn = slim.contrib_framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
#     front_end = end_points['vgg_16/conv4/conv4_3']
#     return front_end, init_fn


# the kernel_size and stride must be 2 integers which represents the directions of height and weight
def deformable_conv2d(features, kernel_size, output_nums, input_nums, stride, index, offset_Trainable=True, mask_Trainable=True):
    kernel_arg_nums = kernel_size[0] * kernel_size[1]
    offset = slim.conv2d(
        features,
        2 * kernel_arg_nums,
        kernel_size,
        stride,
        padding='SAME',
        data_format='NCHW',
        activation_fn=None,
        weights_initializer=tf.zeros_initializer,
        trainable=offset_Trainable,
        scope='offset_parameters_part_' + str(index))

    mask = slim.conv2d(
        features,
        kernel_arg_nums,
        kernel_size,
        stride,
        padding='SAME',
        data_format='NCHW',
        activation_fn=None,
        weights_initializer=tf.zeros_initializer,
        trainable=mask_Trainable,
        scope='mask_parameters_part_' + str(index))

    mask_sigmoid = tf.nn.sigmoid(mask, name='mask_sigmoid_part' + str(index))
    if not mask_Trainable:
        mask_sigmoid = tf.add(mask_sigmoid, 0.5)
    # must set xavier initializer manually, adopt the uniform version
    min = - math.sqrt(6. / (kernel_size[0] * kernel_size[1] * input_nums))
    max = math.sqrt(6. / (kernel_size[0] * kernel_size[1] * input_nums))
    axvier = tf.random_uniform_initializer(minval=min, maxval=max)
    # only use the version of num_groups = 1
    weight = tf.get_variable(
        name='weight_part_' + str(index),
        shape=[output_nums, input_nums, kernel_size[0], kernel_size[1]],
        dtype=tf.float32,
        initializer=axvier
    )

    output = deformable_conv2d_op(
        input=features,
        filter=weight,
        offset=offset,
        mask=mask_sigmoid,
        strides=[1, 1, stride[0], stride[1]],
        num_groups=1,
        deformable_groups=1,
        im2col_step=1,
        no_bias=True,
        padding='SAME',
        data_format='NCHW',
        dilations=[1, 1, 1, 1]
    )
    return output


def DME_inception(features, index, output_nums, input_nums):
    with tf.variable_scope(name_or_scope='deformable_inception_' + str(index)):
        part_1 = deformable_conv2d(features, [3, 3], output_nums, input_nums, [1, 1], index=1)
        # part_1_bias_weights = tf.get_variable(name='part_1_bias_weights', shape=[output_nums], dtype=tf.float32, initializer=tf.zeros_initializer)
        # part_1_bias = tf.nn.bias_add(part_1, part_1_bias_weights, data_format='NCHW', name='part_1_bias')
        part_1_relu = tf.nn.leaky_relu(part_1, name='part_1_leaky_relu')

        part_2 = deformable_conv2d(features, [5, 5], output_nums, input_nums, [1, 1], index=2)
        # part_2_bias_weights = tf.get_variable(name='part_2_bias_weights', shape=[output_nums], dtype=tf.float32, initializer=tf.zeros_initializer)
        # part_2_bias = tf.nn.bias_add(part_2, part_2_bias_weights, data_format='NCHW', name='part_2_bias')
        part_2_relu = tf.nn.leaky_relu(part_2, name='part_2_leaky_relu')

        part_3 = deformable_conv2d(features, [7, 7], output_nums, input_nums, [1, 1], index=3)
        # part_3_bias_weights = tf.get_variable(name='part_3_bias_weights', shape=[output_nums], dtype=tf.float32, initializer=tf.zeros_initializer)
        # part_3_bias = tf.nn.bias_add(part_3, part_3_bias_weights, data_format='NCHW', name='part_3_bias')
        part_3_relu = tf.nn.leaky_relu(part_3, name='part_3_leaky_relu')

    output = tf.concat([part_1_relu, part_2_relu, part_3_relu], axis=1, name='deformable_inception_concat_' + str(index))
    return output


def DME_back_end(features, input_nums):
    features_transpose = tf.transpose(features, [0, 3, 1, 2])
    net_inception_1 = DME_inception(features_transpose, 1, 256, input_nums)
    net_conv_1x1_1 = slim.conv2d(net_inception_1, 256, 1, 1, data_format='NCHW', activation_fn=tf.nn.leaky_relu)
    net_inception_2 = DME_inception(net_conv_1x1_1, 2, 128, 256)
    net_conv_1x1_2 = slim.conv2d(net_inception_2, 128, 1, 1, data_format='NCHW', activation_fn=tf.nn.leaky_relu)
    net_inception_3 = DME_inception(net_conv_1x1_2, 3, 64, 128)
    net_conv_1x1_3 = slim.conv2d(net_inception_3, 1, 1, 1, data_format='NCHW', activation_fn=tf.nn.leaky_relu)
    output = tf.transpose(net_conv_1x1_3, [0, 2, 3, 1], name='ed_map')
    return output


def DME_model(features):
    features = slim.instance_norm(features, activation_fn=tf.nn.relu)
    _, end_points = nets.vgg.vgg_16(features)
    front_end = end_points['vgg_16/conv4/conv4_3']
    feature_map = DME_back_end(front_end, 512)
    return feature_map
