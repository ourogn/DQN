import tensorflow as  tf
import numpy as np
import cv2 as  cv
import random

class agent:
    def __init__(self,K):
        self.k =K
        self.createNet()

    def createNet(self):
        self.input = tf.compat.v1.placeholder(tf.float32, shape=(None, 96, 96, 3), name='input')
        self.tq_value = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='tq_value')
        self.action = tf.compat.v1.placeholder(tf.uint8, name='action')
        self.q_value = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='q_value')
        with tf.compat.v1.variable_scope('weight'):
            self.weights=[
                tf.Variable(tf.random.truncated_normal(shape=[5, 5, 3, 64], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=[5, 5, 64, 64], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=[2, 2, 64, 32], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=(1152, 256), mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=(256, self.k), mean=0, stddev=0.1))

            ]
        with tf.compat.v1.variable_scope('biases'):
            self.biases =[
                tf.Variable(tf.zeros(64)),
                tf.Variable(tf.zeros(64)),
                tf.Variable(tf.zeros(32)),
                tf.Variable(tf.zeros(256)),
                tf.Variable(tf.zeros(self.k))

            ]
        conv1 = tf.nn.relu(tf.nn.conv2d(self.input/255.0,
                             self.weights[0],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
                           + self.biases[0])

        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                        self.weights[1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME')
                           + self.biases[1])

        conv3 = tf.nn.relu(tf.nn.conv2d(conv2,
                                        self.weights[2],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME')
                           + self.biases[2])
        pool = tf.layers.max_pooling2d(conv3,
                                       pool_size=[2,2],
                                       strides=2)
        # pool输出15大小

        flat = tf.layers.flatten(pool)

        fc1 = tf.nn.relu(tf.matmul(flat, self.weights[3]) + self.biases[3])

        fc2 = tf.nn.relu(tf.matmul(fc1, self.weights[4]) + self.biases[4])
        drop = tf.layers.dropout(fc2)

        # 预测值
        self.predict = tf.nn.softmax(drop)
        # q值计算
        action_onehot = tf.one_hot(self.action, self.k)
        self.q_value = tf.reduce_sum(tf.multiply(self.predict, action_onehot), reduction_indices=[1])

        # loss计算
        self.loss = tf.reduce_mean(tf.square(self.tq_value-self.q_value))
        # train
        self.train = tf.compat.v1.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.loss)

    def save(self,session,save_path='abc'):
        saver = tf.compat.v1.train.Saver()
        saver.save(sess=session, save_path=save_path)

    def load(self,session,save_path='abc'):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess=session, save_path=save_path)

    def copyFrom(self,weights,biases,session):
        for i in range(len(self.weights)):
            session.run([self.weights[i].assign(weights[i]),
            self.biases[i].assign(biases[i])])











