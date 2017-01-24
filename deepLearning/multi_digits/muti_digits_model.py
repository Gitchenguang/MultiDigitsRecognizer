# coding: utf-8
import os
import time
import pickle

import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_pickled_data(pickled_file):
    """
    load picked data
    :param pickled_file:
    :return: train_data, train_labels, test_data,
             test_labels, valid_data, valid_labels
    """
    with open(pickled_file, 'rb') as f:
        save = pickle.load(f)
        _train_data = save['train_data']
        _train_labels = save['train_labels']
        _test_data = save['test_data']
        _test_labels = save['test_labels']
        _valid_data = save['valid_data']
        _valid_labels = save['valid_labels']
        del save
        print(_train_data.shape, _train_labels.shape)
        print(_test_data.shape, _test_labels.shape)
        print(_valid_data.shape, _valid_labels.shape)
    return _train_data, _train_labels, _test_data, _test_labels, _valid_data, _valid_labels


def accuracy_func(predicts, labels):
    """
    total accuracy, digit-wise
    :param predicts:
    :param labels:
    :return: float value, precesion
    """
    _predictions = np.argmax(predicts, 2).T
    total_count = 0
    for pre, la in zip(_predictions, labels):
        for i, j in zip(pre.tolist(), la.tolist()):
            if i == j:
                total_count += 1
    # return 100.0 * np.sum(predictions == labels) / predicts.shape[1] / predicts.shape[0]
    return 100.0 * total_count / predicts.shape[1] / predicts.shape[0]


def local_contrast_normalization(input_data, image_shape, threshold=1e-4, radius=7):
    """
    Local Contrast Normalization
    :param input_data: input data
    :param image_shape: image shape
    :param threshold: threshold
    :param radius: redius
    :return: local contrast normalized input data
    """
    # Gaussian filter
    filter_shape = radius, radius, image_shape[3], 1
    filters = gaussian_initializer(filter_shape)
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    convout = tf.nn.conv2d(input_data, filters, [1, 1, 1, 1], 'SAME')
    centered_data = tf.sub(input_data, convout)
    denoms = tf.sqrt(tf.nn.conv2d(tf.square(centered_data), filters, [1, 1, 1, 1], 'SAME'))
    mean = tf.reduce_mean(denoms)
    divisor = tf.maximum(mean, denoms)
    # Divisise step
    new_data = tf.truediv(centered_data, tf.maximum(divisor, threshold))
    return new_data


def gaussian_initializer(kernel_shape):
    """
    initialize the kernel weights
    :param kernel_shape: kernel shape
    :return: tensor
    """
    x = np.zeros(kernel_shape, dtype=float)
    mid = np.floor(kernel_shape[0] / 2.)
    for kernel_idx in range(0, kernel_shape[2]):
        for i in range(0, kernel_shape[0]):
            for j in range(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gaussian(i - mid, j - mid)
    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)


def gaussian(x, y, sigma=3.0):
    """
    gaussian function
    :param x: x value
    :param y: y value
    :param sigma: sigma
    :return: guassian normalized value
    """
    z = 2 * np.pi * sigma ** 2
    return 1. / z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))


class MultiDigits(object):
    """
    Multi Digits Recognition Model
    """
    def __init__(self, picked_file=None, image_size=32, num_labels=11, num_channels=3,
                 batch_size=32, patch_size=5, depth_1=16, depth_2=32, depth_3=64,
                 hidden_num=128, num_hidden1=64
                 ):
        """
        :param picked_file:
        :param image_size:
        :param num_labels:
        :param num_channels:
        :param batch_size:
        :param patch_size:
        :param depth_1:
        :param depth_2:
        :param depth_3:
        :param hidden_num:
        :param num_hidden1:
        """
        self.train_data, self.train_labels, self.test_data, \
            self.test_labels, self.valid_data, self.valid_labels = None, None, None, None, None, None
        if picked_file is not None:
            self.train_data, self.train_labels, self.test_data, \
                self.test_labels, self.valid_data, self.valid_labels = \
                load_pickled_data(picked_file)
        self.train_graph = None
        self.infer_graph = tf.Graph()
        self.image_size = image_size
        self.num_labels = num_labels
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth_1 = depth_1
        self.depth_2 = depth_2
        self.depth_3 = depth_3
        self.hidden_num = hidden_num
        self.num_hidden1 = num_hidden1
        self.shape = [batch_size, image_size, image_size, num_channels]
        self.saver = None
        self.valid_prediction, self.test_prediction = None, None
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=self.shape)
        self.tf_train_labels = None
        self.tf_valid_dataset = None
        self.tf_test_dataset = None
        self.loss = None
        self.optimizer = None
        self.train_prediction = None
        self.save_path = None
        self.infer_saver = None
        self.is_inited = False
        self.valid_acc = None
        self.mini_acc = None
        self.conv_layer1_weights = None
        self.conv_layer1_biases = None
        self.conv_layer2_weights = None
        self.conv_layer2_biases = None
        self.conv_layer2_biases = None
        self.conv_layer3_weights = None
        self.conv_layer3_biases = None
        self.out_weights_len = None
        self.out_biases_len = None
        self.out_weights_1 = None
        self.out_biases_1 = None
        self.out_weights_2 = None
        self.out_weights_2 = None
        self.out_biases_2 = None
        self.out_weights_3 = None
        self.out_biases_3 = None
        self.out_weights_4 = None
        self.out_biases_4 = None
        self.out_weights_5 = None
        self.out_biases_5 = None

    def define_graph(self, keep_pro=0.95, eta=0.05, decay_step=5000, decay_rate=0.95):
        """
        定义图参数
        :param keep_pro: DropOut参数
        :param eta: 学习率
        :param decay_step: 学习率衰减步
        :param decay_rate: 学习率衰减率
        :return:
        """
        self.train_graph = tf.Graph()
        with self.train_graph.as_default():
            # Input Data.
            with tf.name_scope('input'):
                self.tf_train_dataset = tf.placeholder(tf.float32, shape=self.shape)
                self.tf_train_labels = tf.placeholder(tf.int32, shape=(self.batch_size, 6))
                self.tf_valid_dataset = tf.constant(self.valid_data)
                self.tf_test_dataset = tf.constant(self.test_data)
            # init varibales
            # Conv Layers
            with tf.name_scope("conv_weights_1"):
                self.conv_layer1_weights = tf.get_variable('c_1_w', shape=[self.patch_size, self.patch_size,
                                                                           self.num_channels, self.depth_1],
                                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
            with tf.name_scope("conv_biases_1"):
                self.conv_layer1_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_1]), name='c_1_b')
            with tf.name_scope("conv_weights_2"):
                self.conv_layer2_weights = tf.get_variable('c_2_w', shape=[self.patch_size, self.patch_size,
                                                                           self.depth_1, self.depth_2],
                                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
            with tf.name_scope("conv_biases_2"):
                self.conv_layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_2]), name='c_2_b')
            with tf.name_scope("conv_weights_2"):
                self.conv_layer3_weights = tf.get_variable('c_3_w', shape=[self.patch_size, self.patch_size,
                                                                           self.depth_2, self.num_hidden1],
                                                           initializer=tf.contrib.layers.xavier_initializer_conv2d())
            with tf.name_scope("conv_biases_3"):
                self.conv_layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden1]), name='c_3_b')
            # FC layer
                self.fc_layer_weights = tf.get_variable('fc_w', shape=[self.num_hidden1, self.hidden_num],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                self.fc_biases = tf.Variable(tf.constant(1.0, shape=[self.hidden_num]), name='fc_b')
            # Output Layer
            with tf.name_scope("out_w_len"):
                self.out_weights_len = tf.get_variable('o_len', shape=[self.hidden_num, self.num_labels],
                                                       initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_len"):
                self.out_biases_len = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_len'))
            with tf.name_scope("out_w_1"):
                self.out_weights_1 = tf.get_variable('o_1', shape=[self.hidden_num, self.num_labels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_1"):
                self.out_biases_1 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_1'))
            with tf.name_scope("out_w_2"):
                self.out_weights_2 = tf.get_variable('o_2', shape=[self.hidden_num, self.num_labels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_2"):
                self.out_biases_2 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_2'))
            with tf.name_scope("out_w_3"):
                self.out_weights_3 = tf.get_variable('o_3', shape=[self.hidden_num, self.num_labels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_3"):
                self.out_biases_3 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_3'))
            with tf.name_scope("out_w_4"):
                self.out_weights_4 = tf.get_variable('o_4', shape=[self.hidden_num, self.num_labels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_4"):
                self.out_biases_4 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_4'))
            with tf.name_scope("out_w_5"):
                self.out_weights_5 = tf.get_variable('o_5', shape=[self.hidden_num, self.num_labels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope("out_b_5"):
                self.out_biases_5 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_5'))
            # Training computation.
            with tf.name_scope('logits'):
                logitslen, logits1, logits2, logits3, logits4, logits5 = \
                    self._infer(self.tf_train_dataset, keep_pro, self.shape)
            with tf.name_scope('cross_entropy'):
                self.loss = \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitslen, self.tf_train_labels[:, 0])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, self.tf_train_labels[:, 1])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, self.tf_train_labels[:, 2])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, self.tf_train_labels[:, 3])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, self.tf_train_labels[:, 4])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, self.tf_train_labels[:, 5]))
            # Optimizer.
            with tf.name_scope('train'):
                global_step = tf.Variable(0)
                learning_rate = tf.train.exponential_decay(eta, global_step, decay_step, decay_rate)
                self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
            # Predictions of the training, validation, and test data.
            with tf.name_scope('train_prediction'):
                self.train_prediction = tf.pack(list(map(tf.nn.softmax,
                                                         self._infer(self.tf_train_dataset, 1.0, self.shape))))
            with tf.name_scope('valid_prediction'):
                self.valid_prediction = tf.pack(list(map(tf.nn.softmax,
                                                         self._infer(self.tf_valid_dataset, 1.0, self.shape))))
            with tf.name_scope('test_prediction'):
                self.test_prediction = tf.pack(list(map(tf.nn.softmax,
                                                        self._infer(self.tf_test_dataset, 1.0, self.shape))))
            self.saver = tf.train.Saver()
            # create a summary for training loss and validate loss
            tf.scalar_summary('train_loss', self.loss)
            tf.scalar_summary('validate_acc', self.valid_acc)
            tf.scalar_summary('minbat_acc', self.mini_acc)
            self.summary_op = tf.merge_all_summaries()

    def train_model(self, logs_path, save_path=None, save=True, epoch=50000, verbose=True, cross_validate=False):
        """
        训练模型，部署应用的时候不能调用
        :param logs_path: board path
        :param save_path: ckpt数据保存路径
        :param save: 是否保存ckpt数据
        :param epoch: 训练迭代次数
        :param verbose: 显示迭代过程中的中间结果
        :param cross_validate: 是否是交叉验证
        :return: epoch_index, losses, mini_batch_acc, valid_batch_acc, test_acc
        """
        epoch_index = []
        losses = []
        mini_batch_acc = []
        valid_batch_acc = []
        epochs = epoch
        start_time = time.time()
        with tf.Session(graph=self.train_graph) as sess:
            tf.global_variables_initializer().run()
            writer = tf.train.SummaryWriter(logs_path, graph=self.train_graph)

            print('Initialized all variables')
            for e in range(epochs):
                offset = (e * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
                batch_data = self.train_data[offset:(offset + self.batch_size), :, :, :]
                batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                feed_dict = {self.tf_train_dataset: batch_data, self.tf_train_labels: batch_labels}
                _, l, predictions, summary = sess.run([self.optimizer, self.loss, self.train_prediction, self.summary_op], feed_dict=feed_dict)
                if e % 1000 == 0:
                    epoch_index.append(e)
                    self.mini_acc = accuracy_func(predictions, batch_labels)
                    mini_batch_acc.append(self.mini_acc)
                    self.valid_acc = accuracy_func(self.valid_prediction.eval(), self.valid_labels)
                    valid_batch_acc.append(self.valid_acc)
                    losses.append(l)
                    if verbose:
                        print('Minibatch loss at step %d: %f' % (e, l))
                        print('Minibatch accuracy: %.1f%%' % self.mini_acc)
                        if not cross_validate:
                            print('Validation accuracy: %.1f%%' % self.valid_acc)
                writer.add_summary(summary, e)
            test_acc = accuracy_func(self.test_prediction.eval(), self.test_labels)
            print('Test accuracy: %.1f%%' % test_acc)
            if save:
                self.save_path = self.saver.save(sess, save_path)
                print("Model saved in file: %s" % self.save_path)
            end_time = time.time()
            print('train time: %s' % (end_time - start_time))
        return epoch_index, losses, mini_batch_acc, valid_batch_acc, test_acc

    def infer_model(self, input_data, ckpt_path):
        """
        infer input data
        :param input_data: input a instance
        :param ckpt_path: path to the ckpt file
        :return: return result
        """
        infer_graph = tf.Graph()
        with infer_graph.as_default():
            # Input Data.
            tf_infer_data = tf.placeholder(tf.float32, shape=(input_data.shape[0], input_data.shape[1],
                                                              input_data.shape[2], self.num_channels))
            # init varibales
            conv_layer1_weights = tf.get_variable('c_1_w', shape=[self.patch_size, self.patch_size,
                                                                  self.num_channels, self.depth_1],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer1_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_1]), name='c_1_b')
            conv_layer2_weights = tf.get_variable('c_2_w', shape=[self.patch_size, self.patch_size,
                                                                  self.depth_1, self.depth_2],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_2]), name='c_2_b')
            conv_layer3_weights = tf.get_variable('c_3_w', shape=[self.patch_size, self.patch_size,
                                                                  self.depth_2, self.num_hidden1],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden1]), name='c_3_b')
            # Output Layer
            out_weights_len = tf.get_variable('o_len', shape=[self.hidden_num, self.num_labels],
                                              initializer=tf.contrib.layers.xavier_initializer())
            out_biases_len = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_len'))
            out_weights_1 = tf.get_variable('o_1', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_1 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_1'))
            out_weights_2 = tf.get_variable('o_2', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_2 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_2'))
            out_weights_3 = tf.get_variable('o_3', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_3 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_3'))
            out_weights_4 = tf.get_variable('o_4', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_4 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_4'))
            out_weights_5 = tf.get_variable('o_5', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_5 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_5'))

            def infer(data, keep_prob, d_shape):
                # conv layer
                lcn = local_contrast_normalization(data, d_shape)
                conv_1 = tf.nn.conv2d(lcn, conv_layer1_weights, [1, 1, 1, 1], 'VALID', name='c_1')
                conv_1 = tf.nn.relu(conv_1 + conv_layer1_biases)
                conv_1 = tf.nn.local_response_normalization(conv_1)
                pool_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_1')
                conv_2 = tf.nn.conv2d(pool_1, conv_layer2_weights, [1, 1, 1, 1], padding='VALID', name='c_2')
                conv_2 = tf.nn.relu(conv_2 + conv_layer2_biases)
                conv_2 = tf.nn.local_response_normalization(conv_2)
                pool_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_2_')
                conv_3 = tf.nn.conv2d(pool_2, conv_layer3_weights, [1, 1, 1, 1], padding='VALID', name='c_3')
                conv_3 = tf.nn.relu(conv_3 + conv_layer3_biases)
                conv_3 = tf.nn.dropout(conv_3, keep_prob)
                shapes = conv_3.get_shape().as_list()
                hidden = tf.reshape(conv_3, [shapes[0], shapes[1] * shapes[2] * shapes[3]])
                # fc layer
                logits_len = tf.matmul(hidden, out_weights_len) + out_biases_len
                logits_1 = tf.matmul(hidden, out_weights_1) + out_biases_1
                logits_2 = tf.matmul(hidden, out_weights_2) + out_biases_2
                logits_3 = tf.matmul(hidden, out_weights_3) + out_biases_3
                logits_4 = tf.matmul(hidden, out_weights_4) + out_biases_4
                logits_5 = tf.matmul(hidden, out_weights_5) + out_biases_5
                return logits_len, logits_1, logits_2, logits_3, logits_4, logits_5

            logitslen, logits1, logits2, logits3, logits4, logits5 = infer(tf_infer_data, 1.0, self.shape)
            # Predictions
            softmax_result = list(map(tf.nn.softmax, infer(tf_infer_data, 1.0, self.shape)))
            infer_predict = tf.pack(softmax_result)
            prediction = tf.transpose(tf.argmax(infer_predict, 2))
            self.infer_saver = tf.train.Saver()

        with tf.Session(graph=infer_graph) as session:
            self.infer_saver.restore(session, save_path=ckpt_path)
            input_prediction, infer_prediction, l_len, l_1, l_2, l_3, l_4, l_5 = session.run(
                [prediction, infer_predict, logitslen, logits1, logits2, logits3, logits4, logits5],
                feed_dict={tf_infer_data: input_data})
            logits_output = np.array([l_len, l_1, l_2, l_3, l_4, l_5]).reshape((6, 11))
            softmax = lambda data: np.exp(data) / np.sum(np.exp(data))
            df = pd.DataFrame(np.array([softmax(d) for d in logits_output]).T,
                              columns=['length', '1', '2', '3', '4', '5'])
            df['length'] = df['length'].apply(lambda x: '%.10f' % x)
            df['1'] = df['1'].apply(lambda x: '%.10f' % x)
            df['2'] = df['2'].apply(lambda x: '%.10f' % x)
            df['3'] = df['3'].apply(lambda x: '%.10f' % x)
            df['4'] = df['4'].apply(lambda x: '%.10f' % x)
            df['5'] = df['5'].apply(lambda x: '%.10f' % x)
            _index = np.argmax(df['length'])
            df['length'].iloc[_index] = '<b>' + df['length'].iloc[_index] + '</b>'
            _index = np.argmax(df['1'])
            df['1'].iloc[_index] = '<b>' + df['1'].iloc[_index] + '</b>'
            _index = np.argmax(df['2'])
            df['2'].iloc[_index] = '<b>' + df['2'].iloc[_index] + '</b>'
            _index = np.argmax(df['3'])
            df['3'].iloc[_index] = '<b>' + df['3'].iloc[_index] + '</b>'
            _index = np.argmax(df['4'])
            df['4'].iloc[_index] = '<b>' + df['4'].iloc[_index] + '</b>'
            _index = np.argmax(df['5'])
            df['5'].iloc[_index] = '<b>' + df['5'].iloc[_index] + '</b>'

            value = list(map(str, range(10)))
            value.append('no digit')
            df.insert(0, column='softmax', value=value)
            return input_prediction, df

    def infer_data(self, input_data, input_labels, ckpt_path):
        """
        infer input data
        :param input_data: input a instance
        :param ckpt_path: path to the ckpt file
        :param input_labels: input data labels
        :return: return result
        """
        infer_graph = tf.Graph()
        with infer_graph.as_default():
            # Input Data.
            tf_infer_data = tf.placeholder(tf.float32, shape=(input_data.shape[0], input_data.shape[1],
                                                              input_data.shape[2], 1))
            tf_infer_label = tf.placeholder(tf.int32, shape=(input_labels.shape[0], input_labels.shape[1]))
            # init varibales
            conv_layer1_weights = tf.get_variable('c_1_w', shape=[self.patch_size, self.patch_size,
                                                                  self.num_channels, self.depth_1],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer1_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_1]), name='c_1_b')
            conv_layer2_weights = tf.get_variable('c_2_w', shape=[self.patch_size, self.patch_size,
                                                                  self.depth_1, self.depth_2],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.depth_2]), name='c_2_b')
            conv_layer3_weights = tf.get_variable('c_3_w', shape=[self.patch_size, self.patch_size,
                                                                  self.depth_2, self.num_hidden1],
                                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv_layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.num_hidden1]), name='c_3_b')
            # Output Layer
            out_weights_len = tf.get_variable('o_len', shape=[self.hidden_num, self.num_labels],
                                              initializer=tf.contrib.layers.xavier_initializer())
            out_biases_len = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_len'))
            out_weights_1 = tf.get_variable('o_1', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_1 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_1'))
            out_weights_2 = tf.get_variable('o_2', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_2 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_2'))
            out_weights_3 = tf.get_variable('o_3', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_3 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_3'))
            out_weights_4 = tf.get_variable('o_4', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_4 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_4'))
            out_weights_5 = tf.get_variable('o_5', shape=[self.hidden_num, self.num_labels],
                                            initializer=tf.contrib.layers.xavier_initializer())
            out_biases_5 = tf.Variable(tf.constant(1.0, shape=[self.num_labels], name='o_b_5'))

            def infer(data, keep_prob, d_shape):
                # conv layer
                lcn = local_contrast_normalization(data, d_shape)
                conv_1 = tf.nn.conv2d(lcn, conv_layer1_weights, [1, 1, 1, 1], 'VALID', name='c_1')
                conv_1 = tf.nn.relu(conv_1 + conv_layer1_biases)
                conv_1 = tf.nn.local_response_normalization(conv_1)
                pool_1 = tf.nn.max_pool(conv_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_1')
                conv_2 = tf.nn.conv2d(pool_1, conv_layer2_weights, [1, 1, 1, 1], padding='VALID', name='c_2')
                conv_2 = tf.nn.relu(conv_2 + conv_layer2_biases)
                conv_2 = tf.nn.local_response_normalization(conv_2)
                pool_2 = tf.nn.max_pool(conv_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_2_')
                conv_3 = tf.nn.conv2d(pool_2, conv_layer3_weights, [1, 1, 1, 1], padding='VALID', name='c_3')
                conv_3 = tf.nn.relu(conv_3 + conv_layer3_biases)
                conv_3 = tf.nn.dropout(conv_3, keep_prob)
                shapes = conv_3.get_shape().as_list()
                hidden = tf.reshape(conv_3, [shapes[0], shapes[1] * shapes[2] * shapes[3]])
                # fc layer
                logits_len = tf.matmul(hidden, out_weights_len) + out_biases_len
                logits_1 = tf.matmul(hidden, out_weights_1) + out_biases_1
                logits_2 = tf.matmul(hidden, out_weights_2) + out_biases_2
                logits_3 = tf.matmul(hidden, out_weights_3) + out_biases_3
                logits_4 = tf.matmul(hidden, out_weights_4) + out_biases_4
                logits_5 = tf.matmul(hidden, out_weights_5) + out_biases_5
                return logits_len, logits_1, logits_2, logits_3, logits_4, logits_5

            logitslen, logits1, logits2, logits3, logits4, logits5 = infer(tf_infer_data, 1.0, self.shape)

            # Predictions
            _loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitslen, tf_infer_label[:, 0])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_infer_label[:, 1])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_infer_label[:, 2])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_infer_label[:, 3])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_infer_label[:, 4])) + \
                    tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_infer_label[:, 5]))
            infer_predict = tf.pack(list(map(tf.nn.softmax, infer(tf_infer_data, 1.0, self.shape))))
            prediction = tf.transpose(tf.argmax(infer_predict, 2))
            self.infer_saver = tf.train.Saver()

        with tf.Session(graph=infer_graph) as session:
            self.infer_saver.restore(session, save_path=ckpt_path)
            input_prediction, infer_prediction, loss = session.run([prediction, infer_predict, _loss],
                                                                   feed_dict={tf_infer_data: input_data,
                                                                              tf_infer_label: input_labels})
            accuracy = accuracy_func(infer_prediction, input_labels[:])
            return input_prediction, loss, accuracy

    def _infer(self, data, keep_prob, d_shape):
        """
        same as infer_data, for training process
        :param data: data
        :param keep_prob: keep probability for DropOut
        :param d_shape: shape of data
        :return: logits_1, logits_2, logits_3, logits_4, logits_5
        """
        # conv layer
        lcn = local_contrast_normalization(data, d_shape)
        conv_1 = tf.nn.conv2d(lcn, self.conv_layer1_weights, [1, 1, 1, 1], 'VALID', name='c_1')
        conv_1 = tf.nn.relu(conv_1 + self.conv_layer1_biases)
        lrn = tf.nn.local_response_normalization(conv_1)
        pool_1 = tf.nn.max_pool(lrn, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_1')
        conv_2 = tf.nn.conv2d(pool_1, self.conv_layer2_weights, [1, 1, 1, 1], padding='VALID', name='c_2')
        conv_2 = tf.nn.relu(conv_2 + self.conv_layer2_biases)
        lrn = tf.nn.local_response_normalization(conv_2)
        pool_2 = tf.nn.max_pool(lrn, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='p_2_')
        conv_3 = tf.nn.conv2d(pool_2, self.conv_layer3_weights, [1, 1, 1, 1], padding='VALID', name='c_3')
        conv_3 = tf.nn.relu(conv_3 + self.conv_layer3_biases)
        conv_3 = tf.nn.dropout(conv_3, keep_prob)
        shapes = conv_3.get_shape().as_list()
        hidden = tf.reshape(conv_3, [shapes[0], shapes[1] * shapes[2] * shapes[3]])
        # fc layer
        hidden = tf.matmul(hidden, self.fc_layer_weights) + self.fc_biases
        logits_len = tf.matmul(hidden, self.out_weights_len) + self.out_biases_len
        logits_1 = tf.matmul(hidden, self.out_weights_1) + self.out_biases_1
        logits_2 = tf.matmul(hidden, self.out_weights_2) + self.out_biases_2
        logits_3 = tf.matmul(hidden, self.out_weights_3) + self.out_biases_3
        logits_4 = tf.matmul(hidden, self.out_weights_4) + self.out_biases_4
        logits_5 = tf.matmul(hidden, self.out_weights_5) + self.out_biases_5
        return logits_len, logits_1, logits_2, logits_3, logits_4, logits_5
