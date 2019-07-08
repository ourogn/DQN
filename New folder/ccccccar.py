import gym
from Agent import agent
from DQN import DQN
if __name__=='__main__':
    mainAgent = agent(4)
    targetAgent = agent(4)
    dqn = DQN(mainAgent,targetAgent,0.99,0.01,2000,120,400,True)
    dqn.train('CarRacing-v0',True)
