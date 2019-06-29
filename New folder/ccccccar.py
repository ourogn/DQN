import gym
from Agent import agent
from DQN import DQN
if __name__=='__main__':
    mainAgent = agent(6)
    targetAgent = agent(6)
    dqn = DQN(mainAgent,targetAgent,0.9,0.1,2000,32,500)
    dqn.train('CarRacing-v0',False)
