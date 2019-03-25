#coding:utf-8
import tensorflow as tf
import numpy as np
import gym
from gym.spaces import Discrete, Box

### multi-layer perceptron as policy
def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x ,units=size, activation=activation)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def reward_to_go(ep_rews):
    n = len(ep_rews)
    rtgs = np.zeros_like(ep_rews)
    for i in reversed(range(n)):
        rtgs[i] = rtgs[i] + (ep_rews[i+1] if i+1<n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
            epochs=50, batch_size=5000, render=False):

    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    ### Build Policy
    ### 策略网络，输出dim是action space
    obs_ph = tf.placeholder(shape=[None, obs_dim], dtype=tf.float32)
    logits = mlp(obs_ph, sizes=hidden_sizes+[n_acts])

    ### Choose action
    ### 根据策略网络预测的概率，选择一个action
    actions = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)

    ### Loss Function
    ### 损失函数，使得梯度是 
    ###   mean( sum( derivative[log_prob(action_t|state_t)] ) * reward(tau)) 
    ### = mean( sum( derivative[log_prob(action_t|state_t)] * reward(tau) ))
    ### 所以loss是
    ### mean( sum( log_prob(action_t|state_t) * reward(tau) ))

    # act_ph是动作序列，类似于[0, 2, 1]
    act_ph = tf.placeholder(shape=[None,], dtype=tf.int32)
    action_one_hot = tf.one_hot(act_ph, n_acts)

    # reward
    weight_ph = tf.placeholder(shape=[None,], dtype=tf.float32)

    # log_prob
    ### reduce_sum是把action_space的所有动作概率加和，只有一个概率，其他都是0
    log_probs = tf.reduce_sum(tf.nn.log_softmax(logits) * action_one_hot, axis=1)
    ### CartPole这个游戏是一个杆子往下掉，移动底板使它维持平衡，每次移动后没有掉下去reward就是1
    loss = -tf.reduce_mean(log_probs * weight_ph)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = sess.run(actions, {obs_ph: obs.reshape(1,-1)})[0]
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                # batch_weights += [ep_ret] * ep_len
                ### 一个朴素的idea是：当前行为之前的reward不是这个行为的结果；
                ### 所以去除掉：
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        batch_loss, _ = sess.run([loss, train_op],
                                 feed_dict={
                                    obs_ph: np.array(batch_obs),
                                    act_ph: np.array(batch_acts),
                                    weight_ph: np.array(batch_weights)
                                 })
        return batch_loss, batch_rets, batch_lens

    ### training
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)

