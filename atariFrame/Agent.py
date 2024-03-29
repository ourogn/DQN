import tensorflow as  tf
import numpy as np
import cv2 as  cv
import random

class agent:
    def __init__(self,K):
        self.k =K
        self.learningR = 0.00025
        self.createNet()


    def createNet(self):
        self.input = tf.compat.v1.placeholder(tf.float32, shape=(None, 96, 96, 4), name='input')
        self.tq_value = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='tq_value')
        self.action = tf.compat.v1.placeholder(tf.uint8, name='action')
        self.q_value = tf.compat.v1.placeholder(tf.float32, shape=(None,), name='q_value')
        with tf.compat.v1.variable_scope('weight'):
            self.weights=[
                tf.Variable(tf.random.truncated_normal(shape=[16, 16, 4, 16], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=[8, 8, 16, 32], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=[3, 3, 32, 32], mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=(1152, 256), mean=0, stddev=0.1)),
                tf.Variable(tf.random.truncated_normal(shape=(256, self.k), mean=0, stddev=0.1))

            ]
        with tf.compat.v1.variable_scope('biases'):
            self.biases =[
                tf.Variable(tf.zeros(16)),
                tf.Variable(tf.zeros(32)),
                tf.Variable(tf.zeros(32)),
                tf.Variable(tf.zeros(256)),
                tf.Variable(tf.zeros(self.k))

            ]
        #卷积层1
        conv1 = tf.nn.relu(tf.nn.conv2d(self.input/255.0,
                             self.weights[0],
                             strides=[1, 2, 2, 1],
                             padding='SAME')
                           + self.biases[0])
        #卷积层2
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
        fc2 = tf.matmul(fc1, self.weights[4]) + self.biases[4]

        # 预测值
        self.predict = fc2
        # q值计算
        action_onehot = tf.one_hot(self.action, self.k)
        self.q_value = tf.reduce_sum(tf.multiply(self.predict, action_onehot), reduction_indices=[1])

        # loss计算
        self.loss = tf.reduce_mean(tf.square(self.tq_value-self.q_value))
        # train
        self.train = tf.compat.v1.train.RMSPropOptimizer(self.learningR,0.99,0.0,1e-6).minimize(self.loss)

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











