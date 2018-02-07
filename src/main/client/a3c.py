import tensorflow as tf
import numpy as np
import multiprocessing
import threading
import matplotlib.pyplot as plt
import os
import shutil

#from env import Env
# test using gym env
import gym
env = gym.make('CartPole-v0')
N_STATS = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

OUTPUT_GRAPH = True
LOG_DIR = "./log"
N_WORKERS = multiprocessing.cpu_count()

GLOBAL_NET_SCOPE = "Global_Net"
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001
LR_C = 0.001
UPDATE_GLOBAL_ITER = 10
MAX_GLOBAL_EP = 1000

class ACNet:
    def __init__(self, n_actions, n_stats, scope, globalAC=None):
        self.n_actions = n_actions
        self.n_states = n_stats

        if scope == GLOBAL_NET_SCOPE:   # global net
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name="stats")
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, self.n_states], name="stats")
                self.a = tf.placeholder(tf.int32, [None, ], name="actions")
                self.v_target = tf.placeholder(tf.float32, [None, 1], name="Vtarget")
                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name="TD_error")
                with tf.name_scope("c_loss"):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope("a_loss"):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob)*tf.one_hot(self.a, self.n_actions, dtype=tf.float32),
                                             axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td) # stop bp
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob+1e-5),
                                             axis=1, keep_dims=True)
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)

                with tf.name_scope("sync"):
                    with tf.name_scope("pull"):
                        # update global net
                        self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                        self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    with tf.name_scope("push"):
                        # get global net params
                        self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                        self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0.0, 0.01)
        with tf.variable_scope('actor'):
            # relu6 will encourage to learn sparse features earlier
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='l_a')
            a_prob = tf.layers.dense(l_a, self.n_actions, tf.nn.softmax, kernel_initializer=w_init, name='a_prob')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='l_c')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name="v")
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+"/actor")
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+"/critic")
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        weight_probs = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(weight_probs.shape[1]), p=weight_probs.ravel())
        return action

    def save(self):
        saver = tf.train.Saver()
        saver.save(SESS, "./a3c", write_meta_graph=False)

    def load(self):
        saver = tf.train.Saver()
        saver.restore(SESS, "./a3c")

class Worker:
    def __init__(self, name, globalAC):
        self.env = gym.make('CartPole-v0').unwrapped
        self.name = name
        self.AC = ACNet(N_ACTIONS, N_STATS, name, globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                a = self.AC.choose_action(s)
                s_, r, done,info = self.env.step(a)
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                # update global and assign to local net
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis,:]})[0,0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:
                        # combine critic's value
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    feed_dict = {
                        self.AC.s: np.vstack(buffer_s),
                        self.AC.a: np.array(buffer_a),
                        self.AC.v_target: np.vstack(buffer_v_target)
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01*ep_r)
                    print(self.name, "Ep:",GLOBAL_EP, "| Ep_r:%i"%GLOBAL_RUNNING_R[-1])
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name="RMSPropA")
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name="RMSPropC")
        GLOBAL_AC = ACNet(N_ACTIONS, N_STATS, GLOBAL_NET_SCOPE)
        workers = []
        # create workers
        for i in range(N_WORKERS):
            name = "W_%i" % i
            workers.append(Worker(name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    # save the model
    GLOBAL_AC.save()
    print("Saved the model")

    # visualize
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel("step")
    plt.ylabel("Total moving reward")
    plt.show()

    # predict
    AC = ACNet(N_ACTIONS, N_STATS, "local", GLOBAL_AC)
    AC.pull_global()
    env = gym.make('CartPole-v0').unwrapped
    s = env.reset()
    i = 0
    while True:
        env.render()
        a = AC.choose_action(s)
        s_, r, done, info = env.step(a)
        if done:
            break
        s = s_
        i = i+1
    print("steps: %d"%i)