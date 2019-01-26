import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
import cv2


def conv_coordinate_offset_showing(out_h, out_w, kernel_h, kernel_w, offset, stride_h, stride_w, dilation_h, dilation_w, pad_h, pad_w, in_height, in_weight):
    in_coordinate = [[[0, 0] for j in range(kernel_w)] for i in range(kernel_h)]
    x_zero = out_w * stride_w - pad_w
    y_zero = out_h * stride_h - pad_h
    if not(len(offset) == 2 * kernel_h * kernel_w):
        print(' the offset num do not match the kernel weights')
    for i in range(kernel_h):
        for j in range(kernel_w):
            original_w = x_zero + j * dilation_w + int(round(offset[2 * (i * kernel_w + j) + 1][out_h][out_w]))
            original_h = y_zero + i * dilation_h + int(round(offset[2 * (i * kernel_w + j)][out_h][out_w]))
            if (original_h >= 0) and (original_h <= (in_height - 1)):
                in_coordinate[i][j][0] = original_h
            elif original_h < 0:
                in_coordinate[i][j][0] = 0
            elif original_h > in_height - 1:
                in_coordinate[i][j][0] = in_height - 1

            if (original_w >= 0) and (original_w <= (in_weight - 1)):
                in_coordinate[i][j][1] = original_w
            elif original_w < 0:
                in_coordinate[i][j][1] = 0
            elif original_w > in_weight - 1:
                in_coordinate[i][j][1] = in_weight - 1
    return in_coordinate


def wipe_out_repititive_tuples(source):
    dest = []
    for e in source:
        flag = 1
        for d in dest:
            if e[0] == d[0] and e[1] == d[1]:
                flag = 0
                break
        if flag == 1:
            dest.append(e)
    return dest

def dot_image(image, coords_list):
    for i in range(len(coords_list)):
            image[coords_list[i][0]][coords_list[i][1]][0] = 0
            image[coords_list[i][0]][coords_list[i][1]][1] = 0
            image[coords_list[i][0]][coords_list[i][1]][2] = 255
#
# offset = [0 for i in range(2 * 3 * 3)]
# conv_coord_list_10 = []
# conv_coord_list_9 = []
# conv_coord_list_8 = []
# conv_coord_list_7 = []
# conv_coord_list_6 = []
# conv_coord_list_5 = []
# conv_coord_list_4 = []
# conv_coord_list_3 = []
# conv_coord_list_2 = []
# conv_coord_list_1 = []
# pool_coord_list_1 = []
# pool_coord_list_2 = []
# pool_coord_list_3 = []
#
# for i in range(len(front_end_out_coord_list)):
#     h = front_end_out_coord_list[i][0]
#     w = front_end_out_coord_list[i][1]
#     conv_coord_list_10.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 96, 128))
# conv_coord_list_10 = np.reshape(conv_coord_list_10, [-1, 2])
# for i in range(len(conv_coord_list_10)):
#     h = conv_coord_list_10[i][0]
#     w = conv_coord_list_10[i][1]
#     conv_coord_list_9.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 96, 128))
# conv_coord_list_9 = np.reshape(conv_coord_list_9, [-1, 2])
# for i in range(len(conv_coord_list_9)):
#     h = conv_coord_list_9[i][0]
#     w = conv_coord_list_9[i][1]
#     conv_coord_list_8.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 96, 128))
# conv_coord_list_8 = np.reshape(conv_coord_list_8, [-1, 2])
# for i in range(len(conv_coord_list_8)):
#     h = conv_coord_list_8[i][0]
#     w = conv_coord_list_8[i][1]
#     pool_coord_list_3.append(offset_show.conv_coordinate_offset_showing(h, w, 2, 2, [0 for i in range(2 * 2 * 2)], 2, 2, 1, 1, 0, 0, 192, 256))
# pool_coord_list_3 = np.reshape(pool_coord_list_3, [-1, 2])
# pool_coord_list_3 = offset_show.wipe_out_repititive_tuples(pool_coord_list_3)
# for i in range(len(pool_coord_list_3)):
#     h = pool_coord_list_3[i][0]
#     w = pool_coord_list_3[i][1]
#     conv_coord_list_7.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 192, 256))
# conv_coord_list_7 = np.reshape(conv_coord_list_7, [-1, 2])
# for i in range(len(conv_coord_list_7)):
#     h = conv_coord_list_7[i][0]
#     w = conv_coord_list_7[i][1]
#     conv_coord_list_6.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 192, 256))
# conv_coord_list_6 = np.reshape(conv_coord_list_6, [-1, 2])
# for i in range(len(conv_coord_list_6)):
#     h = conv_coord_list_6[i][0]
#     w = conv_coord_list_6[i][1]
#     conv_coord_list_5.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 192, 256))
# conv_coord_list_5 = np.reshape(conv_coord_list_5, [-1, 2])
# for i in range(len(conv_coord_list_5)):
#     h = conv_coord_list_5[i][0]
#     w = conv_coord_list_5[i][1]
#     pool_coord_list_2.append(offset_show.conv_coordinate_offset_showing(h, w, 2, 2, [0 for i in range(2 * 2 * 2)], 2, 2, 1, 1, 0, 0, 384, 512))
# pool_coord_list_2 = np.reshape(pool_coord_list_2, [-1, 2])
# pool_coord_list_2 = offset_show.wipe_out_repititive_tuples(pool_coord_list_2)
# for i in range(len(pool_coord_list_2)):
#     h = pool_coord_list_2[i][0]
#     w = pool_coord_list_2[i][1]
#     conv_coord_list_4.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 384, 512))
# conv_coord_list_4 = np.reshape(conv_coord_list_4, [-1, 2])
# for i in range(len(conv_coord_list_4)):
#     h = conv_coord_list_4[i][0]
#     w = conv_coord_list_4[i][1]
#     conv_coord_list_3.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 384, 512))
# conv_coord_list_3 = np.reshape(conv_coord_list_3, [-1, 2])
# for i in range(len(conv_coord_list_3)):
#     h = conv_coord_list_3[i][0]
#     w = conv_coord_list_3[i][1]
#     pool_coord_list_1.append(offset_show.conv_coordinate_offset_showing(h, w, 2, 2, [0 for i in range(2 * 2 * 2)], 2, 2, 1, 1, 0, 0, 768, 1024))
# pool_coord_list_1 = np.reshape(pool_coord_list_1, [-1, 2])
# pool_coord_list_1 = offset_show.wipe_out_repititive_tuples(pool_coord_list_1)
# for i in range(len(pool_coord_list_1)):
#     h = pool_coord_list_1[i][0]
#     w = pool_coord_list_1[i][1]
#     conv_coord_list_2.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 768, 1024))
# conv_coord_list_2 = np.reshape(conv_coord_list_2, [-1, 2])
# for i in range(len(conv_coord_list_2)):
#     h = conv_coord_list_2[i][0]
#     w = conv_coord_list_2[i][1]
#     conv_coord_list_1.append(offset_show.conv_coordinate_offset_showing(h, w, 3, 3, offset, 1, 1, 1, 1, 1, 1, 768, 1024))
# conv_coord_list_1 = np.reshape(conv_coord_list_1, [-1, 2])
# conv_coord_list_1 = offset_show.wipe_out_repititive_tuples(conv_coord_list_1)