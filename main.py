import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import DME_deformable
import DME
import math
# % matplotlib
# inline
result_output = open("/home/zzn/SANet_implementation-master/result_B_12.13.txt", "w")
image_train_path = "/home/zzn/part_B_final/train_data/images_train.npy"
gt_train_path = "/home/zzn/part_B_final/train_data/gt_train.npy"
image_validate_path = "/home/zzn/part_B_final/train_data/images_validate.npy"
gt_validate_path = "/home/zzn/part_B_final/train_data/gt_validate.npy"
batch_size = 1
epoch = 500
MAE = 19970305
if __name__ == '__main__':
    image_train = np.load(image_train_path)
    gt_train = np.load(gt_train_path)
    image_validate = np.load(image_validate_path)
    gt_validate = np.load(gt_validate_path)

    x = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input")
    y = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="label")

    estimated_density_map, front_end, tmp_inception_1_map, tmp_1x1_1_map, tmp_inception_2_map, tmp_1x1_2_map, tmp_inception_3_map, tmp_1x1_3_map = DME_deformable.DME_model(
        x, 1, 384, 512)
    # estimated_density_map, front_end, tmp_front_end, tmp_inception_value, tmp_1x1_value = DME.DME_model(x)

    estimated_counting = tf.reduce_sum(estimated_density_map, reduction_indices=[1, 2, 3], name='estimated_counting')
    gt_counting = tf.cast(tf.reduce_sum(y, reduction_indices=[1, 2, 3]), tf.float32)

    sum_filter = tf.constant([1. for i in range(64)], dtype=tf.float32, shape=[8, 8, 1, 1])
    gt_map = tf.nn.conv2d(y, sum_filter, [1, 8, 8, 1], padding='SAME')

    loss = tf.squeeze(
        tf.reduce_mean(
            tf.reduce_sum(
                tf.square(estimated_density_map - gt_map),
                reduction_indices=[1, 2, 3]),
            axis=0, name='loss')
        / 2)

    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss=loss, global_step=tf.train.get_global_step())

    AE_batch = tf.abs(tf.subtract(estimated_counting, gt_counting))
    SE_batch = tf.square(tf.subtract(gt_counting, estimated_counting))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # init the Variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        image_train_num = len(image_train)
        step = 0
        for i in range(epoch):
            shuffle_batch = np.random.permutation(image_train_num // batch_size)
            for j in range(image_train_num // batch_size):

                # validate
                if step % 50 == 0:
                    loss_ = []
                    MAE_ = []
                    MSE_ = []
                    ed_map_ = []
                    for k in range(len(image_validate // batch_size)):
                        #                         loss_eval, batch_average_error, batch_square_error, ed_map = sess.run([loss, AE_batch, SE_batch, estimated_density_map],feed_dict={x: image_validate[k:k + batch_size],y: gt_validate[k:k + batch_size]})
                        loss_eval, batch_average_error, batch_square_error, front_g, gt_validate_down_sampling_map, predict_den, gt_counts, pred_counts, tmp_inception_1, tmp_1x1_1, tmp_inception_2, tmp_1x1_2, tmp_inception_3, tmp_1x1_3 = sess.run(
                            [loss, AE_batch, SE_batch, front_end, gt_map, estimated_density_map, gt_counting,
                             estimated_counting, tmp_inception_1_map, tmp_1x1_1_map, tmp_inception_2_map, tmp_1x1_2_map,
                             tmp_inception_3_map, tmp_1x1_3_map],
                            feed_dict={x: image_validate[k:k + batch_size], y: gt_validate[k:k + batch_size]})
                        loss_.append(loss_eval)
                        MAE_.append(batch_average_error)
                        MSE_.append(batch_square_error)
                        #                         ed_map_.append(ed_map)
                        if math.isinf(loss_eval) or math.isnan(loss_eval) or batch_average_error > 1000:
                            #                         if True:
                            figure, ((origin, density_gt, front_ground, pred,
                                      inception_1),
                                     (conv_1x1_1, inception_2, conv_1x1_2, inception_3, conv_1x1_3)) = plt.subplots(2,
                                                                                                                    5,
                                                                                                                    figsize=(
                                                                                                                    20,
                                                                                                                    4))

                            origin.imshow(image_validate[k])
                            origin.set_title('Origin Image')

                            density_gt.imshow(np.squeeze(gt_validate_down_sampling_map), cmap=plt.cm.jet)
                            density_gt.set_title('ground_truth')

                            predict_den = np.squeeze(predict_den)
                            pred.imshow(predict_den, cmap=plt.cm.jet)
                            pred.set_title('back_end')
                            front_ground.imshow(np.squeeze(front_g), cmap=plt.cm.jet)
                            front_ground.set_title('front_end')

                            inception_1.imshow(np.squeeze(tmp_inception_1), cmap=plt.cm.jet)
                            inception_1.set_title('inception_1')
                            conv_1x1_1.imshow(np.squeeze(tmp_1x1_1), cmap=plt.cm.jet)
                            conv_1x1_1.set_title('conv_1x1_1')

                            inception_2.imshow(np.squeeze(tmp_inception_2), cmap=plt.cm.jet)
                            inception_2.set_title('inception_2')
                            conv_1x1_2.imshow(np.squeeze(tmp_1x1_2), cmap=plt.cm.jet)
                            conv_1x1_2.set_title('conv_1x1_2')

                            inception_3.imshow(np.squeeze(tmp_inception_3), cmap=plt.cm.jet)
                            inception_3.set_title('inception_3')
                            conv_1x1_3.imshow(np.squeeze(tmp_1x1_3), cmap=plt.cm.jet)
                            conv_1x1_3.set_title('conv_1x1_3')

                            plt.suptitle("one sample from the validate")
                            plt.show()
                            value_0 = np.sum(front_g)
                            value_1 = np.sum(tmp_inception_1)
                            value_2 = np.sum(tmp_1x1_1)
                            value_3 = np.sum(tmp_inception_2)
                            value_4 = np.sum(tmp_1x1_2)
                            value_5 = np.sum(tmp_inception_3)
                            value_6 = np.sum(tmp_1x1_3)
                            sys.stdout.write(
                                'front_end = {}, inception_1 = {}, conv_1 = {}, inception_2 = {}, conv_2 = {}, inception_3 = {}, conv_3 = {}\n'.format(
                                    value_0, value_1, value_2, value_3, value_4, value_5, value_6))
                            #                             sys.stdout.write('the {}th picture, the loss_origin = {}, the loss_new = {}, gt = {}\n'.format(k, loss_eval,loss_eval_tmp, gt_counts))
                            sys.stdout.flush()
                    # show the validate MAE and MSE values on stdout
                    #                     gt_counts = np.squeeze(gt_counts)
                    #                     pred_counts = np.squeeze(pred_counts)

                    #                     figure, maps = plt.subplots(4, 5, figsize=(20,4))
                    #                     for i_tmp in range(4):
                    #                         for j_tmp in range(5):
                    #                             maps[i_tmp, j_tmp].imshow(np.squeeze(ed_map_[4 * i_tmp + j_tmp]), cmap=plt.cm.jet)
                    #                             maps[i_tmp, j_tmp].set_title('({}, {})'.format(i_tmp, j_tmp))
                    #                     plt.suptitle("all validate samples' predictions")
                    #                     plt.show()

                    loss_ = np.reshape(loss_, [-1])
                    MAE_ = np.reshape(MAE_, [-1])
                    MSE_ = np.reshape(MSE_, [-1])
                    # print(loss_)
                    # print(MAE_)
                    #                     print(MSE_)
                    # calculate the validate loss, validate MAE and validate RMSE
                    validate_loss = np.mean(loss_)
                    validate_MAE = np.mean(MAE_)
                    validate_RMSE = np.sqrt(np.mean(MSE_))

                    # show one of the validate samples
                    figure, ((origin, density_gt, front_ground, pred,
                              inception_1),
                             (conv_1x1_1, inception_2, conv_1x1_2, inception_3, conv_1x1_3)) = plt.subplots(2, 5,
                                                                                                            figsize=(
                                                                                                            20, 4))

                    origin.imshow(image_validate[1])
                    origin.set_title('Origin Image')

                    front_g, gt_validate_down_sampling_map, predict_den, gt_counts, pred_counts, tmp_inception_1, tmp_1x1_1, tmp_inception_2, tmp_1x1_2, tmp_inception_3, tmp_1x1_3 = sess.run(
                        [front_end, gt_map, estimated_density_map, gt_counting, estimated_counting, tmp_inception_1_map,
                         tmp_1x1_1_map, tmp_inception_2_map, tmp_1x1_2_map, tmp_inception_3_map, tmp_1x1_3_map],
                        feed_dict={x: image_validate[1:2], y: gt_validate[1:2]})

                    density_gt.imshow(np.squeeze(gt_validate_down_sampling_map), cmap=plt.cm.jet)
                    density_gt.set_title('ground_truth')

                    predict_den = np.squeeze(predict_den)
                    pred.imshow(predict_den, cmap=plt.cm.jet)
                    pred.set_title('back_end')
                    front_ground.imshow(np.squeeze(front_g), cmap=plt.cm.jet)
                    front_ground.set_title('front_end')

                    inception_1.imshow(np.squeeze(tmp_inception_1), cmap=plt.cm.jet)
                    inception_1.set_title('inception_1')
                    conv_1x1_1.imshow(np.squeeze(tmp_1x1_1), cmap=plt.cm.jet)
                    conv_1x1_1.set_title('conv_1x1_1')

                    inception_2.imshow(np.squeeze(tmp_inception_2), cmap=plt.cm.jet)
                    inception_2.set_title('inception_2')
                    conv_1x1_2.imshow(np.squeeze(tmp_1x1_2), cmap=plt.cm.jet)
                    conv_1x1_2.set_title('conv_1x1_2')

                    inception_3.imshow(np.squeeze(tmp_inception_3), cmap=plt.cm.jet)
                    inception_3.set_title('inception_3')
                    conv_1x1_3.imshow(np.squeeze(tmp_1x1_3), cmap=plt.cm.jet)
                    conv_1x1_3.set_title('conv_1x1_3')

                    plt.suptitle("one sample from the validate")
                    plt.show()

                    # show the validate MAE and MSE values on stdout
                    gt_counts = np.squeeze(gt_counts)
                    pred_counts = np.squeeze(pred_counts)
                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}\n'.format(gt_counts, pred_counts))
                    sys.stdout.write(
                        'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, i + 1, validate_loss,
                                                                                          validate_MAE, validate_RMSE))

                    sys.stdout.flush()

                    # save model
                    if MAE > validate_MAE and not (math.isnan(validate_MAE) or math.isinf(validate_MAE)):
                        MAE = validate_MAE
                        saver.save(sess, './checkpoint_dir/MyModel_deformable')

                #                 train
                start = (shuffle_batch[j] * batch_size) % image_train_num
                end = min(start + batch_size, image_train_num)
                sess.run(train_op, feed_dict={x: image_train[start:end], y: gt_train[start:end]})
                step = step + 1

