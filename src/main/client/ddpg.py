import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

# hyper parameters
MAX_EPISODES = 2000
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9 # reward discount
TAU = 0.01  # target net replacement weight
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
N_LAYER = 30

RENDER = False
ENV_NAME = "CartPole-v0"

class DDPG:
    def __init__(self, a_dim, s_dim):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.sess = tf.Session()

        # memory
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2+1+1), dtype=np.float32)
        self.pointer = 0

        # input
        self.S = tf.placeholder(tf.float32, [None, self.s_dim], name="S")
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], name="S_")
        self.R = tf.placeholder(tf.float32, [None, 1], name="R")

        # construct net
        with tf.variable_scope("Actor"):
            self.a = self._build_actor(self.S, 'eval', trainable=True)  # update frequently
            a_ = self._build_actor(self.S_, 'target', trainable=False)  # update slowly
        with tf.variable_scope("Critic"):
            q = self._build_critic(self.S, self.a, 'eval', trainable=True)  # update frequently
            q_ = self._build_critic(self.S_, a_, 'target', trainable=False) # update slowly

        # parameters
        self.ae_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(at, (1-TAU)*at+TAU*ae), tf.assign(ct, (1-TAU)*ct+TAU*ce)]
                             for at,ae,ct,ce in zip(self.at_param,self.ae_param,self.ct_param,self.ce_param)]

        q_target = self.R + GAMMA*q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_param)

        a_loss = - tf.reduce_mean(q)    # max the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_param)

        # initialize
        self.sess.run(tf.global_variables_initializer())

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, N_LAYER, activation=tf.nn.relu, name="layer1", trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=None, name="action", trainable=trainable)
            return tf.nn.softmax(a, name="action_prob")

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            # share variables (according to the actor's layer number)
            w1_s = tf.get_variable('w1_s', [self.s_dim, N_LAYER], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, N_LAYER], trainable=trainable)
            b1 = tf.get_variable('b1', [1,N_LAYER], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, name="QValue", trainable=trainable)

    def choose_action(self, s):
        weight_probs = self.sess.run(self.a, feed_dict={self.S: s[np.newaxis, :]})
        action = np.random.choice(range(weight_probs.shape[1]), p=weight_probs.ravel())
        return action

    def learn(self):
        # target net update
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        dt = self.memory[indices,:]
        ds = dt[:,:self.s_dim]
        da = dt[:,self.s_dim:self.s_dim+self.a_dim]
        dr = dt[:,self.s_dim+self.a_dim:self.s_dim+self.a_dim+1]
        ds_ = dt[:,-self.s_dim:]

        self.sess.run(self.atrain, {self.S: ds})
        self.sess.run(self.ctrain, {self.S: ds, self.a: da, self.R: dr, self.S_: ds_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    ddpg = DDPG(a_dim, s_dim)

    r_list = []
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            a = ddpg.choose_action(s)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS-1 or done:
                print('Episode:',i,' Reward: %i' % int(ep_reward))
                r_list.append(int(ep_reward))
                #if i > MAX_EPISODES/2:  RENDER = True
                break

    # visualize
    plt.plot(np.arange(len(r_list)), r_list)
    plt.xlabel("step")
    plt.ylabel("Total moving reward")
    plt.show()