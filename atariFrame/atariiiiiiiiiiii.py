import gym
from Agent import agent
from DQN import DQN
if __name__=='__main__':
    mainAgent = agent(18)
    targetAgent = agent(18)
    dqn = DQN(mainAgent,targetAgent,0.99,0.1,200,32,64,True)
    dqn.train('Boxing-v0',True)
