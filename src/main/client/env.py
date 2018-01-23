# 封装和仿真系统交互的接口

from client import get_batch_response
import numpy as np

class Env:
    def __init__(self, n):
        self.state_dim = n*3
        self.action_dim = n
        data = '{"n":'+str(n)+',"action":0}'
        ret = get_batch_response(data, "start")
        self.city = ret['city']
        for x in self.city:
            x.append(0)
        self.state = ret['state']
        self.actions = [x for x in range(1,n)]

    def step(self, action):
        data = '{"n":1,"action":'+str(action)+'}'
        ret = get_batch_response(data, "run")
        temp = ret['state'][1:self.action_dim+1]
        for x in temp:
            if x<0:
                continue
            else:
                self.city[x][2] = 1
        s = np.reshape(self.city, self.action_dim*3).tolist()
        r = ret['distance']
        done = True if ret['done']==1 else False
        return s,r,done

    def reset(self, n):
        self.__init__(n)
        s = np.reshape(self.city, self.action_dim * 3).tolist()
        return s

    def render(self):
        pass

    def sample_action(self):
        if self.actions == []:
            return 0
        a = np.random.choice(self.actions)
        self.actions.remove(a)
        return a

if __name__ == "__main__":
    env = Env(5)
    print(env.city)
    print(env.state)

    done = False
    while not done:
        a = env.sample_action()
        s,r,done = env.step(a)
        print(s,r,done)