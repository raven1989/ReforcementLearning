import gym

# env = gym.make("MountainCar-v0")
# env = gym.make("SpaceInvaders-v0")
env = gym.make("CartPole-v0")
print(env.observation_space.shape)
obs = env.reset()
done = False
for t in range(1000):
    # env.render()
    last_obs = obs
    act = env.action_space.sample()
    obs, rew, done, info = env.step(act)
    print("T:{}\tObs:{}\tAct:{}\tRew:{}\tinfo:{}".format(t, last_obs, act, rew, info))
    if done:
        print("Episode finished after {} timesteps.".format(t))
        break

env.close()
