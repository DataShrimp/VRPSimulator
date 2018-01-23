# 强化学习主框架

from env import Env
from agent import Agent
import numpy as np

N = 10
MAX_EPISODES = 10
MAX_EP_STEPS = N

EPSILON = 1.0
HIDDEN_SIZE = 20
LEARN_RATE = 0.02

env = Env(N)
agent = Agent(env.action_dim, env.state_dim, HIDDEN_SIZE, LEARN_RATE)

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
            agent.learn()
            break
    agent.save()
    EPSILON -= EPSILON/MAX_EPISODES

if __name__ == "__main__":
    print("hi")