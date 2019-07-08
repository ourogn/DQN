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
                 min_exp,isDDQN):
        self.agent =agent
        self.target_agent = target_agent
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps = max_eps
        self.exp_size = exp_size
        self.batch_size = batch_size
        self.exp_buff = []
        self.min_exp = min_exp
        self.isDDQN = isDDQN

    def change_eps(self):
        self.eps =max(self.eps-(self.max_eps-self.min_eps)/200000,
                      self.min_eps)
    def train(self,name,isShow):
        env = gym.envs.make(name)

        self.sess = tf.compat.v1.Session()
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        self.allStep=0
        # 读取数据
        # self.agent.load(self.sess)
        state = env.reset()
        state = processImg(state)
        x = np.zeros((96, 96, 4))
        for i in range(4):
            x[:, :, i] = state
        n_x = x
        epiode_rewards = np.zeros(500)
        last_ten_rewards = []
        for i in range(self.min_exp):
            action = random.randint(0,self.agent.k-1)

            next_state, reward, done, _ = env.step(actionCov(action))
            next_state = processImg(next_state)
            for i in range(3):
                n_x[:, :, i] = n_x[:,:,i+1]
            n_x[:,:,3] = next_state

            self.exp_buff.append((x, action, reward, n_x,done))
            if done:
                state = env.reset()
                state =processImg(state)
                for i in range(4):
                    x[:, :, i] = state
                n_x = x
            else:
                x = n_x;
        for q in range(2000):

            state = env.reset()
            image_rgb = state
            state = processImg(state)
            x = np.zeros((96,96,4))
            for i in range(4):
                x[:,:,i] = state
            n_x = x
            num_steps = 0
            act_step =0
            allReward = 0
            done = False
            while (not done) and allReward >= 0 and np.mean(image_rgb[:,:,1]) < 184:
                if self.allStep % 1600 == 0:
                    self.target_agent.copyFrom(weights=self.agent.weights,
                                               session=self.sess,
                                               biases=self.agent.biases)
                if act_step%8==0:
                    action = self.getAction(x[np.newaxis,:,:,:])
                if self.eps==self.min_eps:
                    self.eps==self.max_eps
                    self.agent.learningR=self.agent.learningR*10
                next_state, reward, done, _ = env.step(actionCov(action))
                image_rgb = next_state
                if isShow:
                    env.render()
                next_state = processImg(next_state)
                for i in range(3):
                    n_x[:, :, i] = n_x[:, :, i + 1]
                n_x[:, :, 3] = next_state
                allReward += reward
                reward=reward/10
                if len(self.exp_buff)==self.exp_size:
                    self.exp_buff.pop(0)
                if act_step%8==0:
                    self.exp_buff.append((x, action, reward, n_x,done))
                    loss = self.learn()
                    print(loss, action)
                act_step+=1
                x=n_x
                self.change_eps()
                num_steps += 1
                self.allStep+=1


            print(num_steps, allReward)
            epiode_rewards[q] = allReward
            last_avg = epiode_rewards[max(0, q - 10):q + 1].mean()
            last_ten_rewards.append(last_avg)


            if (q % 20 == 10):
                self.agent.save(self.sess)
                #save_eps_steps(self)
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
            return random.randint(0 ,self.agent.k-1)
        actions=self.sess.run(self.agent.predict,feed_dict={self.agent.input:state})
        return np.argmax(actions)
    def learn(self):
        # 算target_Q
        samples = random.sample(self.exp_buff,self.batch_size)
        states,actions,rewards,next_states,dones = map(np.array,zip(*samples))
        if self.isDDQN:
            mainAct = self.sess.run(self.agent.predict,feed_dict={self.agent.input:next_states})
            mainAct = np.argmax(mainAct,axis=1)
            t_q = self.sess.run(self.target_agent.q_value,feed_dict={self.target_agent.input:next_states,
                                                                     self.target_agent.action:mainAct})
        else:
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
    '''x = np.empty([1, 96, 96, 3])
    x[0, :, :, :] = img
    return x'''
    s = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return s


def actionCov(action):
    if action == 0:
        return [0.0, 0.0, 0.0]
    elif action == 1:
        return [-0.6, 0.05, 0.0]
    elif action == 2:
        return [0.6, 0.05, 0.0]
    elif action == 3:
        return [0.0, 0.3,0.0]


def save_eps_steps(dqn):
    re = {'eps':dqn.eps,'step':dqn.allStep}
    file = open("eps_steps.txt", "w")
    for k, v in re.items():
        line = k.encode("utf-8") + "\t" + str(v) + "\n"
        file.write(line)
    file.close()
