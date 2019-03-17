import gym
from pyglet.gl import *
from spinup.utils.test_policy import load_policy, run_policy

_, get_action = load_policy("/search/odin/tensorflow/chenlu/spinningup/data/installtest/installtest_s0", 'last', True)
env = gym.make('LunarLander-v2')
# print(type(env))
# env.reset()
# env.render()
run_policy(env, get_action)

env.close()
