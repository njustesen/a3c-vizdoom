import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from tensorflow.python.ops.gen_array_ops import _const

import a3c_helpers
import a3c_network
import a3c_constants as constants

from helper import *
from vizdoom import *

from random import choice
from time import sleep
from time import time

class Worker():
    def __init__(self, game, name, s_size, a_size, trainer, model_path, global_episodes):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_" + str(self.number))

        self.bots = constants.BOTS

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = a3c_network.AC_Network(s_size, a_size, self.name, trainer)
        self.update_local_ops = a3c_helpers.update_target_graph('global', self.name)

        # The Below code is related to setting up the Doom environment
        #game.set_doom_scenario_path("scenarios/basic_cig.wad")  # This corresponds to the simple task we will pose our agent
        #game.load_config("scenarios/basic_cig.cfg")

        game.set_doom_scenario_path("scenarios/cig.wad")  # This corresponds to the simple task we will pose our agent
        game.load_config("scenarios/cig.cfg")
        game.set_doom_map("map02")
        game.add_game_args("+name AI +colorset 0")
        game.add_game_args("-host 1 -deathmatch +timelimit 2.0 "
                           "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")

        game.set_screen_resolution(ScreenResolution.RES_160X120)
        game.set_screen_format(ScreenFormat.GRAY8)
        game.set_render_hud(constants.RENDER_HUD)
        game.set_render_crosshair(constants.RENDER_CROSSHAIR)
        game.set_render_weapon(constants.RENDER_WEAPON)
        game.set_render_decals(constants.RENDER_DECALS)
        game.set_render_particles(constants.RENDER_PARTICLES)
        #game.add_available_button(Button.MOVE_LEFT)
        #game.add_available_button(Button.MOVE_RIGHT)
        #game.add_available_button(Button.TURN_LEFT)
        #game.add_available_button(Button.TURN_RIGHT)
        #game.add_available_button(Button.ATTACK)
        game.add_available_game_variable(GameVariable.AMMO0)
        game.add_available_game_variable(GameVariable.AMMO1)
        game.add_available_game_variable(GameVariable.AMMO2)
        game.add_available_game_variable(GameVariable.AMMO3)
        game.add_available_game_variable(GameVariable.AMMO4)
        game.add_available_game_variable(GameVariable.AMMO5)
        game.add_available_game_variable(GameVariable.AMMO6)
        game.add_available_game_variable(GameVariable.AMMO7)
        game.add_available_game_variable(GameVariable.AMMO8)
        game.add_available_game_variable(GameVariable.AMMO9)
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.add_available_game_variable(GameVariable.ON_GROUND)
        game.add_available_game_variable(GameVariable.KILLCOUNT)
        game.add_available_game_variable(GameVariable.DEATHCOUNT)
        game.add_available_game_variable(GameVariable.ARMOR)
        game.add_available_game_variable(GameVariable.FRAGCOUNT)
        game.add_available_game_variable(GameVariable.HEALTH)
        #game.set_episode_timeout(300)
        game.set_episode_timeout(constants.EPISODE_TIMEOUT)
        game.set_episode_start_time(constants.EPISODE_START_TIME)
        game.set_window_visible(constants.WINDOW_VISIBLE)
        game.set_sound_enabled(constants.SOUND_ENABLED)
        #game.set_living_reward(-1)
        game.set_mode(Mode.PLAYER)
        # game.set_mode(Mode.ASYNC_PLAYER)
        game.init()

        game.send_game_command("removebots")
        for i in range(self.bots):
            game.send_game_command("addbot")

        self.actions = self.actions = np.identity(a_size, dtype=bool).tolist()
        # End Doom set-up
        self.env = game

        # Reward function - Shaped
        #self.reward_function = [0.01,0.01,0,-1,1,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
        # Reward function - Frags
        self.goals = [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = a3c_helpers.discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = a3c_helpers.discount(advantages, gamma)

        goals = []
        for i in range(len(observations)):
            goals.append(self.goals)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.input_image: np.vstack(observations),
                     self.local_AC.input_goals: goals,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: self.batch_rnn_state[0],
                     self.local_AC.state_in[1]: self.batch_rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, self.batch_rnn_state, _ = sess.run([self.local_AC.value_loss,
                                                                     self.local_AC.policy_loss,
                                                                     self.local_AC.entropy,
                                                                     self.local_AC.grad_norms,
                                                                     self.local_AC.var_norms,
                                                                     self.local_AC.state_out,
                                                                     self.local_AC.apply_grads],
                                                                    feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, max_episode_length, gamma, sess, coord, saver):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_frames = []
                episode_reward = 0
                episode_step_count = 0
                d = False

                self.env.new_episode()
                position_history = []
                last_vars = a3c_helpers.get_vizdoom_vars(self.env, position_history)
                s = self.env.get_state().screen_buffer
                episode_frames.append(s)
                s = a3c_helpers.process_frame(s)
                rnn_state = self.local_AC.state_init
                self.batch_rnn_state = rnn_state
                while self.env.is_episode_finished() == False:

                    # Respawn if dead
                    if self.env.is_player_dead():
                        self.env.respawn_player()

                    position = a3c_helpers.get_position(self.env)
                    position_history.append(position)

                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={self.local_AC.input_image: [s],
                                   self.local_AC.input_goals: [self.goals],
                                   self.local_AC.state_in[0]: rnn_state[0],
                                   self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)
                    r = self.env.make_action(self.actions[a], constants.FRAME_SKIP) / 100.0

                    # Evaluate reward based on vars and reward function
                    vars = a3c_helpers.get_vizdoom_vars(self.env, position_history)
                    delta_vars = np.subtract(vars, last_vars)
                    rewards = np.multiply(self.goals, delta_vars)
                    r = np.sum(rewards)
                    last_vars = vars

                    d = self.env.is_episode_finished()
                    if d == False:
                        s1 = self.env.get_state().screen_buffer
                        episode_frames.append(s1)
                        s1 = a3c_helpers.process_frame(s1)
                    else:
                        s1 = s

                    episode_buffer.append([s, a, r, s1, d, v[0, 0]])
                    episode_values.append(v[0, 0])

                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if len(episode_buffer) == constants.BATCH_SIZE and d != True and episode_step_count != max_episode_length - 1:
                        # Since we don't know what the true final return is, we "bootstrap" from our current
                        # value estimation.
                        v1 = sess.run(self.local_AC.value,
                                      feed_dict={self.local_AC.input_image: [s],
                                                 self.local_AC.input_goals: [self.goals],
                                                 self.local_AC.state_in[0]: rnn_state[0],
                                                 self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
                        v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, v1)
                        episode_buffer = []
                        sess.run(self.update_local_ops)
                    if d == True:
                        break

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                # Update the network using the episode buffer at the end of the episode.
                if len(episode_buffer) != 0:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % 5 == 0 and episode_count != 0:
                    if self.name == 'worker_0' and episode_count % 25 == 0:
                        time_per_step = 0.05
                        images = np.array(episode_frames)
                        make_gif(images, './frames/image' + str(episode_count) + '.gif',
                                 duration=len(images) * time_per_step, true_image=True, salience=False)
                    if episode_count % 250 == 0 and self.name == 'worker_0':
                        saver.save(sess, self.model_path + '/model-' + str(episode_count) + '.cptk')
                        print("Saved Model")

                    mean_reward = np.mean(self.episode_rewards[-5:])
                    mean_length = np.mean(self.episode_lengths[-5:])
                    mean_value = np.mean(self.episode_mean_values[-5:])
                    summary = tf.Summary()
                    summary.value.add(tag='Perf/Reward', simple_value=float(mean_reward))
                    summary.value.add(tag='Perf/Length', simple_value=float(mean_length))
                    summary.value.add(tag='Perf/Value', simple_value=float(mean_value))
                    summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
                    summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
                    summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
                    summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    summary.value.add(tag='Losses/Var Norm', simple_value=float(v_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()

                    print("EPISODE: " + str(episode_count) + " | MEAN REWARD: " + str(mean_reward) + " | MEAN VALUE: " + str(mean_value))

                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1