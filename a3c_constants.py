# To launch tensorboard in dir tensorboard --logdir=./
from vizdoom import *
import numpy as np

# TRAINING CONSTANTS
GAMMA = 0.99
FRAME_SIZE = [160,120,1]
OBSERVATION_SIZE = np.product(FRAME_SIZE)
ACTIONS_SIZE = 7
LOAD_MODEL = False
MODEL_PATH = './model'
POSITION_DECAY = 0.9
BATCH_SIZE = 32
FRAME_SKIP = 4
VAR_SIZE = 16
MAX_THREADS = 1

# NOVELTY
EVENTS = 16
EVENT_CAPACITY = 25
MEAN_EVENT_CLIP = 0.1
MIN_REWARD = 0
MAX_REWARD = 100

# VIZDOOM CONSTANTS
SCREEN_RESOLUTION = ScreenResolution.RES_160X120
BOTS = 7
RENDER_HUD = False
RENDER_CROSSHAIR = False
RENDER_WEAPON = True
RENDER_DECALS = False
RENDER_PARTICLES = False
EPISODE_TIMEOUT = 252
EPISODE_START_TIME = 10
WINDOW_VISIBLE = False
SOUND_ENABLED = False
