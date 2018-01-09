# Python
import tensorflow as tf
import numpy as np

'''
values = tf.placeholder(shape=[None, 4], dtype=tf.float32)
target_v = tf.placeholder(shape=[None, 4], dtype=tf.float32)
sub = target_v - values
squared = tf.square(sub)
value_loss = 0.5 * tf.reduce_sum(squared)

sess = tf.Session()
sub, squared, out = sess.run([sub, squared, value_loss], feed_dict={values:[[1,2,3,4],[1,2,3,4]], target_v:[np.ones(4), np.ones(4)]})


print(sub)
print(squared)
print(out)
'''

import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import a3c_helpers
import a3c_constants

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

n = 3
v = 2

class AC_Network():
    def __init__(self, scope, trainer):
        with tf.variable_scope(scope):
            self.input_vars = tf.placeholder(shape=[None, 2], dtype=tf.float32)

            self.hidden = slim.fully_connected(self.input_vars, 1024, activation_fn=tf.nn.elu)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(self.hidden, a3c_constants.ACTIONS_SIZE,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=a3c_helpers.normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.values = slim.fully_connected(self.hidden, n,
                                              activation_fn=None,
                                              weights_initializer=a3c_helpers.normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                #self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
                #self.actions_onehot = tf.one_hot(self.actions, a3c_constants.ACTIONS_SIZE, dtype=tf.float32)
                self.target_v = tf.placeholder(shape=[None, n], dtype=tf.float32)
                #self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                #self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.sub = self.target_v - self.values
                self.squared = tf.square(self.target_v - self.values)
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - self.values))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy))
                #self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
                #self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.01

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.value_loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
net = AC_Network('net', trainer)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    feed_dict = {
        net.target_v: [[0,0,0],[1,1,1],[2,2,2]],
        net.input_vars: [[1,2],[4,5],[6,7]]
    }

    values, sub, squared, loss = sess.run([net.values, net.sub, net.squared, net.value_loss], feed_dict=feed_dict)

    print(values)
    print(sub)
    print(squared)
    print(loss)