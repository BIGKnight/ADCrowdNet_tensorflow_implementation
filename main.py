import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import DME_deformable
import DME

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

    estimated_density_map, front_end, tmp_front_end, tmp_inception_value, tmp_1x1_value = DME_deformable.DME_model(x, 1, 384, 512)
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
                    for k in range(len(image_validate // batch_size)):
                        loss_eval, batch_average_error, batch_square_error = sess.run([loss, AE_batch, SE_batch],
                                                                                      feed_dict={x: image_validate[
                                                                                                    k:k + batch_size],
                                                                                                 y: gt_validate[
                                                                                                    k:k + batch_size]})
                        loss_.append(loss_eval)
                        MAE_.append(batch_average_error)
                        MSE_.append(batch_square_error)
                    #                         print(k, batch_average_error)

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
                    figure, ((origin, density_gt), (pred, front_ground)) = plt.subplots(2, 2, figsize=(20, 4))
                    origin.imshow(image_validate[1])
                    origin.set_title('Origin Image')

                    front_g, gt_validate_down_sampling_map, predict_den, gt_counts, pred_counts, front_end_value, inception_value, conv1x1_value = sess.run(
                        [front_end, gt_map, estimated_density_map, gt_counting, estimated_counting, tmp_front_end, tmp_inception_value, tmp_1x1_value],
                        feed_dict={x: image_validate[1:2], y: gt_validate[1:2]})

                    density_gt.imshow(np.squeeze(gt_validate_down_sampling_map), cmap=plt.cm.jet)
                    density_gt.set_title('ground_truth')

                    predict_den = np.squeeze(predict_den)
                    pred.imshow(predict_den, cmap=plt.cm.jet)
                    pred.set_title('back_end')
                    front_ground.imshow(np.squeeze(front_g), cmap=plt.cm.jet)
                    front_ground.set_title('front_end')

                    plt.suptitle("one sample from the validate")
                    plt.show()

                    # show the validate MAE and MSE values on stdout
                    gt_counts = np.squeeze(gt_counts)
                    pred_counts = np.squeeze(pred_counts)

                    sys.stdout.write(
                        'The gt counts of the above sample:{}, and the pred counts:{}, vgg:{}, inception_1:{}, conv1x1:{}\n'.format(gt_counts, pred_counts, front_end_value, inception_value, conv1x1_value))
                    sys.stdout.write(
                        'In step {}, epoch {}, with loss {}, MAE = {}, MSE = {}\n'.format(step, i + 1, validate_loss,
                                                                                          validate_MAE, validate_RMSE))
                    sys.stdout.flush()

                    # save model
                    if MAE > validate_MAE:
                        MAE = validate_MAE
                        saver.save(sess, './checkpoint_dir/MyModel_deformable')

                # train
                start = (shuffle_batch[j] * batch_size) % image_train_num
                end = min(start + batch_size, image_train_num)
                sess.run(train_op, feed_dict={x: image_train[start:end], y: gt_train[start:end]})
                step = step + 1

