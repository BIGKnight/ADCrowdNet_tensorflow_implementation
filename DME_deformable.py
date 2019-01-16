
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
# from deformable_conv2d_op.deformable_conv2d import deformable_conv2d_op
import os.path as osp
from tensorflow.python.framework import ops

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
    return [data_grad, filter_grad, offset_grad * 0.1, mask_grad * 0.1]# List of 4 Tensor, since we have 4 input


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
def deformable_conv2d(features, kernel_size, output_nums, input_nums, stride, batch_size, image_height, image_weight, index):
    kernel_arg_nums = kernel_size[0] * kernel_size[1]

    args = slim.conv2d(
        features,
        3 * kernel_arg_nums,
        kernel_size,
        stride,
        padding='SAME',
        data_format='NCHW',
        weights_initializer=tf.zeros_initializer,
        scope='offset_mask_parameters_part_' + str(index))

    offset = tf.slice(
        args,
        [0, 0, 0, 0],
        [batch_size, 2 * kernel_arg_nums, image_height, image_weight],
        name='offest_part_' + str(index)
    )

    mask_origin = tf.slice(
        args,
        [0, 2 * kernel_arg_nums, 0, 0],
        [batch_size, kernel_arg_nums, image_height, image_weight],
        name='mask_origin_part_' + str(index)
    )

    mask_sigmoid = tf.sigmoid(mask_origin, name='mask_sigmoid_part_' + str(index))

    # only use the version of num_groups = 1
    weight = tf.get_variable(
        # name='deformable_kernel_inception_' + str(index) + '_' + str(kernel_size[0]) + 'x' + str(kernel_size[1]),
        name='weight_part_' + str(index),
        shape=[output_nums, input_nums, kernel_size[0], kernel_size[1]],
        initializer=tf.random_uniform_initializer(maxval=1)
    )

    # print(features, kernel_size, output_nums, input_nums, stride, batch_size, image_height, image_weight, index)

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


def DME_inception(features, index, output_nums, input_nums, batch_size, image_height, image_weight):
    with tf.variable_scope(name_or_scope='deformable_inception_' + str(index)):
        part_1 = deformable_conv2d(features, [3, 3], output_nums, input_nums, [1, 1], batch_size, image_height=image_height, image_weight=image_weight, index=1)
        part_1_relu = tf.nn.relu(part_1, name='part_1_relu')

        part_2 = deformable_conv2d(features, [5, 5], output_nums, input_nums, [1, 1], batch_size, image_height=image_height, image_weight=image_weight, index=2)
        part_2_relu = tf.nn.relu(part_2, name='part_2_relu')

        part_3 = deformable_conv2d(features, [7, 7], output_nums, input_nums, [1, 1], batch_size, image_height=image_height, image_weight=image_weight, index=3)
        part_3_relu = tf.nn.relu(part_3, name='part_3_relu')

    output = tf.concat([part_1_relu, part_2_relu, part_3_relu], axis=1, name='deformable_inception_concat_' + str(index))
    return output


def DME_back_end(features, input_nums, batch_size, image_height, image_weight):
    features_transpose = tf.transpose(features, [0, 3, 1, 2])
    net_inception_1 = DME_inception(features_transpose, 1, 256, input_nums, batch_size, image_height=image_height, image_weight=image_weight)
    net_inception_1 = slim.conv2d(net_inception_1, 256, 1, 1, data_format='NCHW')
    net_inception_2 = DME_inception(net_inception_1, 2, 128, 256, batch_size, image_height=image_height, image_weight=image_weight)
    net_inception_2 = slim.conv2d(net_inception_2, 128, 1, 1, data_format='NCHW')
    net_inception_3 = DME_inception(net_inception_2, 3, 64, 128, batch_size, image_height=image_height, image_weight=image_weight)
    net_inception_4 = slim.conv2d(net_inception_3, 1, 1, 1, data_format='NCHW')
    output = tf.transpose(net_inception_4, [0, 2, 3, 1])
    return output


def DME_model(features, batch_size, image_height, image_weight):
    features = slim.instance_norm(features, activation_fn=tf.nn.relu)
    _, end_points = nets.vgg.vgg_16(features)
    front_end = end_points['vgg_16/conv4/conv4_3']
    image_height = int(image_height / 8)
    image_weight = int(image_weight / 8)
    feature_map = DME_back_end(front_end, 512, batch_size, image_height=image_height, image_weight=image_weight)

    front_g = slim.conv2d(front_end, 1, 1)

    return feature_map, front_g
