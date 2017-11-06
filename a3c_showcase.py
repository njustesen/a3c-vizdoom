import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import a3c_helpers
from a3c_network import AC_Network
from a3c_worker import Worker
import a3c_constants as constants
import ga

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

gamma = constants.GAMMA # discount rate for advantage estimation and reward discounting
s_size = constants.OBSERVATION_SIZE # Observations are greyscale frames of 84 * 84 * 1
a_size = constants.ACTIONS_SIZE # Agents actions are found in the config
load_model = constants.LOAD_MODEL
model_path = constants.MODEL_PATH

tf.reset_default_graph()

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Create a directory to save episode playback gifs to
if not os.path.exists('./frames'):
    os.makedirs('./frames')

with tf.device("/cpu:0"):
    global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
    trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
    master_network = AC_Network('global', None)  # Generate global network
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    print('Loading Model...')
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    worker = Worker(DoomGame(), 0, trainer, model_path, global_episodes, ga)
    for i in range(10):
        worker.showcase(sess)