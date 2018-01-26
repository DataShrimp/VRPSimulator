# 强化学习网络设计

import tensorflow as tf
import numpy as np

class Agent:
    def __init__(self, n_actions, n_features, n_fc1, lr, output_graph=False):
        self.sess = tf.Session()
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_fc1 = n_fc1
        self.lr = lr
        self.ep_observes, self.ep_actions, self.ep_rewards = [], [], []

        self.is_visual = output_graph
        self._build_network()
        self.sess.run(tf.global_variables_initializer())

        # tensorboard: 0.0.0.0:6006
        if output_graph:
            self.writer = tf.summary.FileWriter("logs/", self.sess.graph)
            self.merge = tf.summary.merge_all()


    def _build_network(self):
        with tf.name_scope("inputs"):
            self.observe_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.n_features], name="observes")
            self.action_holder = tf.placeholder(dtype=tf.int32, shape=[None], name="actions")
            self.reward_holder = tf.placeholder(dtype=tf.float32, shape=[None], name="rewards")

        fc1 = tf.layers.dense(self.observe_holder, self.n_fc1, activation=tf.nn.relu, name="fc1")
        all_act = tf.layers.dense(fc1, self.n_actions, activation=None, name="fc2")
        self.out = tf.nn.softmax(all_act, name="act_prob")

        # tensorboard
        if self.is_visual:
            tf.summary.histogram("fc1", fc1)
            tf.summary.histogram("fc2", all_act)

        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.action_holder)
            loss = tf.reduce_mean(neg_log_prob*self.reward_holder)

            # tensorboard
            if self.is_visual:
                tf.summary.scalar("loss", loss)

        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

    def choose_action(self, observes):
        observes = np.array(observes)
        weight_probs = self.sess.run(self.out, feed_dict={self.observe_holder: observes[np.newaxis,:]})
        action = np.random.choice(range(weight_probs.shape[1]), p=weight_probs.ravel())
        return action

    def learn(self, step = -1):
        if self.is_visual and step>=0:
            _, ret = self.sess.run([self.train_op, self.merge], feed_dict={
                self.observe_holder: np.vstack(self.ep_observes),
                self.action_holder: np.array(self.ep_actions),
                self.reward_holder: np.array(self.ep_rewards)
            })
            self.ep_observes, self.ep_actions, self.ep_rewards = [], [], []
            self.writer.add_summary(ret, step)
        else:
            self.sess.run(self.train_op, feed_dict={
                self.observe_holder: np.vstack(self.ep_observes),
                self.action_holder: np.array(self.ep_actions),
                self.reward_holder: np.array(self.ep_rewards)
            })
            self.ep_observes, self.ep_actions, self.ep_rewards = [], [], []

    def store_transition(self, s, a, r):
        self.ep_observes.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, "./model", write_meta_graph=False)

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model")
