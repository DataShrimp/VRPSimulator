# 强化学习主框架

from env import Env
from agent import Agent
import numpy as np

N = 10
TRAIN = 1

MAX_EPISODES = 10000
MAX_EP_STEPS = N

EPSILON = 1.0
HIDDEN_SIZE = 50
LEARN_RATE = 0.01

env = Env(N)
agent = Agent(env.action_dim, env.state_dim, HIDDEN_SIZE, LEARN_RATE, output_graph=True)

# train
if TRAIN:
    for i in range(MAX_EPISODES):
        s = env.reset(N)

        for j in range(MAX_EP_STEPS):
            if np.random.random() < EPSILON:
                a = env.sample_action()
            else:
                a = agent.choose_action(s)
                if a in env.actions:
                    env.actions.remove(a)
                else:
                    a = env.sample_action()

            s_, r, done = env.step(a)
            agent.store_transition(s, a, r)
            s = s_

            if done:
                agent.learn(i)
                break
        agent.save()
        EPSILON -= EPSILON/MAX_EPISODES
        if i%100 == 0:
            print("Episode process, step:{0}".format(i))

# test
# TODO：在服务器端控制台看仿真结果与理论最优结果之间的偏差
if not TRAIN:
    agent.load()
    s = env.reset(N)
    for i in range(N):
        a = agent.choose_action(s)
        while a not in env.actions and a>0:
            print("prediction failed, action: {0}".format(a))
            a = agent.choose_action(s)
        if a>0:
            env.actions.remove(a)
        s, r, done = env.step(a)
        print("Action: {0}, Reward: {1}".format(a, r))

if __name__ == "__main__":
    print("done")