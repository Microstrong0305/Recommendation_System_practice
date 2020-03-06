import os
import sys
import tensorflow as tf
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import numpy as np
import argparse
from FM_binaryClass.util import *
from sklearn.metrics import *


class FM(object):
    """初始化成员变量"""

    def __init__(self, feature_size, fm_v_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param):
        # 特征向量长度
        self.feature_size = feature_size
        # fm_v_size 向量的长度
        self.fm_v_size = fm_v_size
        # 损失函数
        self.loss_fuc = loss_fuc
        # 优化方法
        self.train_optimizer = train_optimizer
        # 学习率
        self.learning_rate = learning_rate
        # 正则类型
        self.reg_type = reg_type
        # 正则因子
        self.reg_param = reg_param
        # aglobal_step
        self.glabal_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def train(self, batch_data):
        """ 1 定义输入数据 """
        with tf.name_scope('input_data'):
            # 标签：[batch_size, 1]
            labels = batch_data['labels']
            # 用户特征向量：[batch_size, feature_size]
            dense_vector = tf.reshape(batch_data['dense_vector'],
                                      shape=[-1, self.feature_size, 1])  # None * feature_size * 1
            print("%s: %s" % ("dense_vector", dense_vector))
            print("%s: %s" % ("labels", labels))

        """ 2 定义网络输出 """
        with tf.name_scope("FM_Comput_Score"):
            # FM参数，生成或者获取W V
            with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
                self.FM_W = tf.get_variable(name='fm_w', shape=[self.feature_size, 1],
                                            initializer=tf.glorot_normal_initializer())
                self.FM_V = tf.get_variable(name='fm_v', shape=[self.feature_size, self.fm_v_size],
                                            initializer=tf.glorot_normal_initializer())
                self.FM_B = tf.Variable(tf.constant(0.0), dtype=tf.float32, name="fm_bias")  # W0
            print("%s: %s" % ("FM_W", self.FM_W))
            print("%s: %s" % ("FM_V", self.FM_V))
            print("%s: %s" % ("FM_B", self.FM_B))

            # ---------- w * x----------
            Y_first = tf.reduce_sum(tf.multiply(self.FM_W, dense_vector), 2)  # None * F
            print("%s: %s" % ("Y_first", Y_first))

            # ---------- Vij * Vij* Xij ---------------
            embeddings = tf.multiply(self.FM_V, dense_vector)  # None * V * X
            # sum_square part
            summed_features_emb = tf.reduce_sum(embeddings, 1)  # sum(v*x)
            summed_features_emb_square = tf.square(summed_features_emb)  # (sum(v*x))^2

            # square_sum part
            squared_features_emb = tf.square(embeddings)  # (v*x)^2
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # sum((v*x)^2)

            # second order
            Y_second = 0.5 * tf.subtract(summed_features_emb_square,
                                         squared_sum_features_emb)  # 0.5*((sum(v*x))^2 - sum((v*x)^2))
            print("%s: %s" % ("Y_second", Y_second))

            # out = W * X + Vij * Vij* Xij
            FM_out_lay1 = tf.concat([Y_first, Y_second], axis=1)
            Y_Out = tf.reduce_sum(FM_out_lay1, 1)
            # out = out + bias
            y_d = tf.reshape(Y_Out, shape=[-1])
            Y_bias = self.FM_B * tf.ones_like(y_d, dtype=tf.float32)  # Y_bias
            Y_Out = tf.add(Y_Out, Y_bias, name='Y_Out')
            print("%s: %s" % ("Y_bias", Y_bias))
            print("%s: %s" % ("Y_Out", Y_Out))
            # ---------- score ----------
            score = tf.nn.sigmoid(Y_Out, name='score')
            score = tf.reshape(score, shape=[-1, 1])
            print("%s: %s" % ("score", score))

        """ 3 定义损失函数和AUC指标 """
        with tf.name_scope("loss"):
            # loss：Squared_error，Cross_entropy ,FTLR
            if self.reg_type == 'l1_reg':
                regularization = tf.contrib.layers.l1_regularizer(self.reg_param)(self.FM_W)
            elif self.reg_type == 'l2_reg':
                regularization = self.reg_param * tf.nn.l2_loss(self.FM_W)
            else:
                regularization = self.reg_param * tf.nn.l2_loss(self.FM_W)

            if self.loss_fuc == 'Squared_error':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            elif self.loss_fuc == 'Cross_entropy':
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.reshape(Y_Out, [-1]),
                                                                              labels=tf.reshape(labels,
                                                                                                [-1]))) + regularization
            elif self.loss_fuc == 'FTLR':
                loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - score), reduction_indices=[1])) + regularization
            # AUC
            auc = tf.metrics.auc(labels, score)
            print("%s: %s" % ("labels", labels))
            # w为0的比例,w的平均值
            w_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(self.FM_W) <= 1.0e-5))
            w_avg = tf.reduce_mean(self.FM_W)
            v_zero_ratio = tf.reduce_mean(tf.to_float(tf.abs(self.FM_V) <= 1.0e-5))
            v_avg = tf.reduce_mean(self.FM_V)

        """ 4 设定optimizer """
        with tf.name_scope("optimizer"):
            # ------bulid optimizer------
            with tf.variable_scope("Optimizer", reuse=tf.AUTO_REUSE):
                if self.train_optimizer == 'Adam':
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                       epsilon=1e-8)
                elif self.train_optimizer == 'Adagrad':
                    optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                          initial_accumulator_value=1e-8)
                elif self.train_optimizer == 'Momentum':
                    optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95)
                elif self.train_optimizer == 'ftrl':
                    optimizer = tf.train.FtrlOptimizer(self.learning_rate)
                train_step = optimizer.minimize(loss, global_step=self.global_step)

        """5 设定summary，以便在Tensorboard里进行可视化 """
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("accumulate_auc", auc[0])
            tf.summary.scalar("w_avg", w_avg)
            tf.summary.scalar("w_zero_ratio", w_zero_ratio)
            tf.summary.scalar("v_avg", v_avg)
            tf.summary.scalar("v_zero_ratio", v_zero_ratio)
            tf.summary.histogram("FM_W", self.FM_W)
            tf.summary.histogram("FM_V", self.FM_V)
            # 好几个summary，所以这里要merge_all
            summary_op = tf.summary.merge_all()

        """6 返回结果 """
        return Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, v_zero_ratio, v_avg, labels, score, summary_op


def train_model(sess, model, epochs=1000, print_every=50):
    """training model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)

    # get number of batches
    num_batches = len(x_train) // batch_size + 1

    for e in range(epochs):
        num_samples = 0
        losses = []
        for ibatch in range(num_batches):
            # batch_size data
            batch_x, batch_y = next(batch_gen)
            batch_y = np.array(batch_y).astype(np.float32)
            actual_batch_size = len(batch_y)
            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_x,
                         model.y: batch_y,
                         model.keep_prob: 1.0}

            loss, accuracy, summary, global_step, _ = sess.run([model.loss, model.accuracy,
                                                                merged, model.global_step,
                                                                model.train_op], feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                saver.save(sess, "checkpoints/model", global_step=global_step)
        # print loss of one epoch
        total_loss = np.sum(losses) / num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e + 1))


def test_model(sess, model, print_every=50):
    """training model"""
    # get testing data, iterable
    all_ids = []
    all_clicks = []
    # get number of batches
    num_batches = len(y_test) // batch_size + 1

    for ibatch in range(num_batches):
        # batch_size data
        batch_x, batch_y = next(test_batch_gen)
        actual_batch_size = len(batch_y)
        # create a feed dictionary for this15162 batch
        feed_dict = {model.X: batch_x,
                     model.keep_prob: 1}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        y_out_prob = np.array(y_out_prob[0])

        batch_clicks = np.argmax(y_out_prob, axis=1)

        batch_y = np.argmax(batch_y, axis=1)

        print(confusion_matrix(batch_y, batch_clicks))
        ibatch += 1
        if ibatch % print_every == 0:
            logging.info("Iteration {0} has finished".format(ibatch))


def shuffle_list(data):
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]


def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my Factorization Machine")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # get mode (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str, default='train')
    args = parser.parse_args()
    mode = args.mode

    # length of representation
    x_train, y_train, x_test, y_test = load_dataset()

    # initialize the model
    feature_size = 530
    fm_v_size = 20
    loss_fuc = 'Cross_entropy'
    train_optimizer = 'Adam'
    learning_rate = 0.01
    reg_type = 'l2_reg'
    reg_param = 0.000
    log_path = '/data/log/FM_Cross_entropy_L2_0_20180816_01'
    batch_size = 128

    # initialize FM model
    batch_gen = batch_generator([x_train, y_train], batch_size)
    test_batch_gen = batch_generator([x_test, y_test], batch_size)
    model = FM(feature_size, fm_v_size, loss_fuc, train_optimizer, learning_rate, reg_type, reg_param)

    Y_Out, score, regularization, loss, auc, train_step, w_zero_ratio, w_avg, v_zero_ratio, v_avg, labels, score, summary_op = model.train(
        batch_gen)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        batch_cnt = 0
        # 选定可视化存储目录
        writer = tf.summary.FileWriter(log_path, sess.graph)
        try:
            while True:
                batch_cnt = batch_cnt + 1
                a, b, c, d, e, summary = sess.run([loss, auc, w_zero_ratio, w_avg, train_step, summary_op])
                if batch_cnt % 50 == 0 or batch_cnt <= 10:
                    y, p = sess.run([labels, score])
                    if y.sum() > 0.0:
                        batch_auc = metrics.roc_auc_score(y, p)
                    else:
                        batch_auc = 0.0
                    print(
                        "batch: {} loss: {:.4f} accumulate_auc: {:.4f} batch_auc: {:.4f} w_zero_ratio: {:.4f} w_avg: {:.4f}".format(
                            batch_cnt, a, b[0], batch_auc, c, d))
                    writer.add_summary(summary, batch_cnt)
        except tf.errors.OutOfRangeError:
            print("3、Train end of dataset")
