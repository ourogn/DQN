import tensorflow as  tf
import gym
import cv2 as  cv
import numpy as np
import random


class DQN:
    def __init__(self,agent,
                 target_agent,
                 max_eps,
                 min_eps,
                 exp_size,
                 batch_size,
                 min_exp):
        self.agent =agent
        self.target_agent = target_agent
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps = max_eps
        self.exp_size = exp_size
        self.batch_size = batch_size
        self.exp_buff = []
        self.min_exp = min_exp

    def change_eps(self):
        self.eps =max(self.eps-(self.max_eps-self.min_eps)/500000,
                      self.min_eps)
    def train(self,name,isShow):
        env = gym.envs.make(name)

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        self.allStep=0
        self.agent.load(self.sess)
        state = env.reset()
        state = processImg(state)
        epiode_rewards = np.zeros(500)
        last_ten_rewards = []
        for i in range(self.min_exp):
            action = np.random.randint(0,4)

            next_state, reward, done, _ = env.step(actionCov(action))
            next_state = processImg(next_state)

            self.exp_buff.append((state[0], action, reward, next_state[0],done))
            if done:
                state = env.reset()
                state =processImg(state)
            else:
                state = next_state
        for q in range(500):

            state = env.reset()
            state = processImg(state)
            num_steps = 0

            allReward = 0
            done = False
            while (not done) and allReward >= 0:
                if self.allStep % 10000 == 0:
                    self.target_agent.copyFrom(weights=self.agent.weights,
                                               session=self.sess,
                                               biases=self.agent.biases)

                action = self.getAction(state)

                next_state, reward, done, _ = env.step(actionCov(action))
                if isShow:
                    env.render()
                next_state = processImg(next_state)
                allReward += reward
                if len(self.exp_buff)==self.exp_size:
                    self.exp_buff.pop(0)
                self.exp_buff.append((state[0], action, reward, next_state[0],done))


                # learning
                #loss = self.learn()



                state = next_state
                self.change_eps()
                num_steps += 1
                self.allStep+=1
                #if num_steps%100==0:
                    #print(loss,action)

            print(num_steps, allReward)
            epiode_rewards[q] = allReward
            last_avg = epiode_rewards[max(0, q - 10):q + 1].mean()
            last_ten_rewards.append(last_avg)


            if (q % 20 == 10):
                self.agent.save(self.sess)
                save_eps_steps(self)
            if(q%10 ==0):
                import matplotlib.pyplot as plt
                plt.plot(last_ten_rewards)
                plt.xlabel('episodes')
                plt.ylabel('Average RW')
                plt.savefig("RW.png")

        import matplotlib.pyplot as plt
        plt.plot(last_ten_rewards)
        plt.xlabel('episodes')
        plt.ylabel('Average RW')
        plt.savefig("RW.png")

        env.close()


    def getAction(self,state):
        if random.random()<self.eps:
            return random.randint(0 ,self.agent.k)
        actions=self.sess.run(self.agent.predict,feed_dict={self.agent.input:state})
        return np.argmax(actions)
    def learn(self):
        # 算target_Q
        samples = random.sample(self.exp_buff,self.batch_size)
        states,actions,rewards,next_states,dones = map(np.array,zip(*samples))
        t_qs = self.sess.run(self.target_agent.predict,feed_dict={self.target_agent.input:next_states})
        t_q = np.amax(t_qs,axis=1)
        targets = rewards+np.invert(dones).astype(np.float32)*0.9*t_q
        # 更新神经网络
        loss,_ = self.sess.run(
            [self.agent.loss,self.agent.train],
            feed_dict={
                self.agent.input:states,
                self.agent.tq_value:targets,
                self.agent.action:actions
            }
        )
        return loss


def processImg(img):
    x = np.empty([1, 96, 96, 3])
    x[0, :, :, :] = img
    return x
    s = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(s, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(s, contours, -1, (0, 0, 255), 3)
    x =np.empty([1, 96, 96, 1])
    x[0,:,:,0] =s
    return x


def actionCov(action):
    if action == 0:
        return [-1, 1, 0]
    elif action == 1:
        return [1, 1, 0.4]
    elif action == 2:
        return [-1, 1, 0.4]
    elif action == 3:
        return [1, 1,0]
    elif action ==4 :
        return [0,1,0]
    else:
        return [0,1,0.4]

def save_eps_steps(dqn):
    re = {'eps':dqn.eps,'step':dqn.allStep}
    file = open("eps_steps.txt", "w")
    for k, v in re.items():
        line = k.encode("utf-8") + "\t" + str(v) + "\n"
        file.write(line)
    file.close()