import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
import math
import a3c_constants

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

# Processes Doom screen image to produce cropped and resized image.
def process_frame(frame):
    #s = frame[10:-10,30:-30]
    s = scipy.misc.imresize(frame,a3c_constants.FRAME_SIZE)
    '''
    if a3c_constants.FRAME_SIZE[2] == 1:
        s = np.reshape(s,[np.prod(s.shape)]) / 255.0
    else:
        s = s / 255.0
    '''
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0
    return s

# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def movement_reward(position_history):
    reward = 0
    idx = 0
    last_position = None
    for p in reversed(position_history):
        if last_position is not None:
            distance = math.sqrt((p[0] - last_position[0]) ** 2 + (p[1] - last_position[1]) ** 2) / 10
            decay = a3c_constants.POSITION_DECAY ** idx
            reward += decay * distance
            idx += 1
        last_position = p
    return reward / max(1,idx)

def meters_walked(position_history):
    meters = 0
    last_position = None
    for p in position_history:
        if last_position is None:
            last_position = p
        else:
            distance = math.sqrt((p[0] - last_position[0]) ** 2 + (p[1] - last_position[1]) ** 2) / 100
            if distance > 1:
                meters += 1
                last_position = p

    return meters

def get_position(vizdoom):
    return [vizdoom.get_game_variable(GameVariable.POSITION_X), vizdoom.get_game_variable(GameVariable.POSITION_Y)]

def get_vizdoom_vars(vizdoom, position_history):
    vars = []
    vars.append(meters_walked(position_history))                     # 0
    vars.append(vizdoom.get_game_variable(GameVariable.HEALTH))      # 1
    vars.append(vizdoom.get_game_variable(GameVariable.ARMOR))       # 2
    vars.append(vizdoom.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO))       # 3
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON0))       # 4
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON1))       # 5
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON2))       # 6
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON3))       # 7
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON4))       # 8
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON5))       # 9
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON6))       # 10
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON7))       # 11
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON8))       # 12
    vars.append(vizdoom.get_game_variable(GameVariable.WEAPON9))    # 13
    vars.append(vizdoom.get_game_variable(GameVariable.KILLCOUNT))   # 14
    vars.append(vizdoom.get_game_variable(GameVariable.DEATHCOUNT))  # 15

    return np.array(vars)


def get_events(vars, last_vars):
    events = np.zeros(a3c_constants.EVENTS)

    # If died -> no event
    if vars[15] > last_vars[15]:
        return events

    # 0. Movement
    if vars[0] > last_vars[0]:
        events[0] = 1;

    # 1. Health increase
    if vars[1] > last_vars[1]:
        events[1] = 1;

    # 2. Armor increase
    if vars[2] > last_vars[2]:
        events[2] = 1;

    # 3. Ammo decrease
    if vars[3] < last_vars[3]:
        events[3] = 1;

    # 4. Ammo increase
    if vars[3] > last_vars[3]:
        events[4] = 1;

    # 5-14. Weapons 0-9
    for i in range(4,14):
        if vars[i] > last_vars[i]:
            events[i+1] = 1;

    # 15. Kill increase
    if vars[14] > last_vars[14]:
        events[15] = 1;

    return events



