import gym
from Agent import agent
from DQN import DQN
if __name__=='__main__':
    mainAgent = agent(4)
    targetAgent = agent(4)
    dqn = DQN(mainAgent,targetAgent,0.9,0.1,200,20,40)
    dqn.train('CarRacing-v0',True)
