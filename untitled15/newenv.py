import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import pandas as pd
import random
from random import choice
from collections import  Counter

class NewEnv4:

    def __init__(self, process_time, job_size, capacity):
        #self.num_machines = num_machines
        self.num_jobs = len(job_size)
        self.process_time = process_time
        self.job_size = job_size
        self.capacity = capacity
        self.buffer1 = np.array([])
        self.max_state_value = np.sum(self.process_time)
        self.seed()
        #-------------------------
        # -------------------------
        z, j, min_batch = 0, 0, 1
        batch = [0]
        # process_time1=sorted(process_time,reverse=True)
        df = pd.DataFrame([process_time, job_size])
        df1 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
        df2 = df1.sort_values(by=0, ascending=False, inplace=True)
        job_size_temp = df1[1].values
        process_time_temp = df1[0].values
        print(process_time_temp)
        for i in range(len(job_size_temp)):
            j += job_size_temp[i]
            if j <= capacity:#capacity=30
                if batch[-1] < process_time_temp[i]:
                    batch[-1] = process_time_temp[i]
            else:
                batch.append(process_time_temp[i])#每一批的最大时间[c1,c2,c3,c4....]
                j = job_size_temp[i]#累计后的装载量
                min_batch += 1
        print("启发式算法批处理时间：", batch)
        print("启发式算法完工时间：", sum(batch))
        print("批数下界:", 1 + int(np.sum(job_size) / self.capacity))
        print("总时间为：", self.max_state_value)
        print("差距为：", self.max_state_value-sum(batch))

        # ---------------------------
        self.num_states = 2 * self.num_jobs
        self.min_batch=self.num_jobs
        #self.num_states = self.num_states + 2
        #int(np.sum(job_size)/capacity)+1

        self.max_job_processing_time = np.max(self.process_time)
        self.action_space = spaces.Discrete(8)
        print(self.action_space)
        high1 = np.append(np.array([self.max_job_processing_time] * self.num_jobs), np.array([self.capacity] * self.num_jobs))
        self.observation_space = spaces.Box(np.zeros(self.num_states), high1, dtype=np.float64)
        #print(self.observation_space)
        self.state = np.array([0.0 for j in range(self.num_states)])
        #print("state",self.state)
        #print(self.state.shape)
        self.steps_beyond_done = 0
        self._max_episode_steps = self.num_jobs
        #self.lb=np.sum(self.process_time)/self.num_machines
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print(action)
        job_index = 0
        bacth_index = 0
        strat = int(self.min_batch)
        #print("strat",strat)
        reward=0
        if action == 0:
            temp=[]
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.process_time[i])
                else:
                    temp.append(-1)
            job_index = np.argmax(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1#最接近capacity
            #print("tem1:",tem1)
            #print("temp:",temp)
            bacth_index = np.where(temp1 > 0, temp1, 10000).argmin()
            #print("bacth_index",bacth_index)
            #print("buffer1",self.buffer1)

        if action == 1:
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.process_time[i])
                else:
                    temp.append(-1)
            job_index = np.argmax(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 < 0, temp1, 10000).argmax()
            #print("bacth_index",bacth_index)


        if action == 2:
            #job_index = random.choice([i for i in range(self.num_jobs) if i not in self.buffer1])
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.job_size[i])
                else:
                    temp.append(100000)
            job_index = np.argmin(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            #print("temp1",temp1)
            bacth_index = np.where(temp1 > 0, temp1, 10000).argmin()#塞的最满的

        if action == 3:
            temp = []
            #job_index = random.choice([i for i in range(self.num_jobs) if i not in list(self.buffer1)])
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.job_size[i])
                else:
                    temp.append(100000)
            job_index = np.argmin(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 < 0, temp1, 10000).argmax()

        if action == 4:
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.process_time[i] / self.job_size[i])
                else:
                    temp.append(-1)
            job_index = np.argmax(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1 = self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 < 0, temp1, 10000).argmax()

        if action == 5:
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(self.process_time[i] / self.job_size[i])
                else:
                    temp.append(-1)
            job_index = np.argmax(np.array(temp))
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 > 0, temp1, 10000).argmin()

        if action == 6:
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(i)
                else:
                    pass
            job_index = random.choice(temp)
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 > 0, temp1, 10000).argmin()

        if action == 7:
            temp = []
            for i in range(int(self.num_jobs)):
                if i not in list(self.buffer1):
                    temp.append(i)
                else:
                    pass
            job_index = random.choice(temp)
            self.buffer1 = np.append(self.buffer1, job_index)

            tem1=self.state[strat:]
            temp1 = self.capacity-self.job_size[job_index]-tem1
            # print(temp)
            bacth_index = np.where(temp1 < 0, temp1, 10000).argmax()

        # if action == 6:
        #     job_index = random.choice([i for i in range(len(self.job_size))
        #                                if i not in self.buffer1])
        #     self.buffer1 = np.append(self.buffer1, job_index)
        #     self.buffer2 = np.append(self.buffer2, self.process_time[job_index])
        #     self.buffer3 = np.append(self.buffer3, self.job_size[job_index])
        #
        #     strat = int(self.min_batch)
        #     tem1 = self.state[strat:]
        #     temp1 = self.capacity - self.job_size[job_index] - np.array(tem1[::-1])
        #     # print(temp)
        #     bacth_index = np.where(temp1 < 0, temp1, 10000).argmax()
        #
        # if action == 7:
        #     job_index = random.choice([i for i in range(len(self.job_size))
        #                                if i not in self.buffer1])
        #     self.buffer1 = np.append(self.buffer1, job_index)
        #     self.buffer2 = np.append(self.buffer2, self.process_time[job_index])
        #     self.buffer3 = np.append(self.buffer3, self.job_size[job_index])
        #
        #     strat = int(self.min_batch)
        #     tem1=self.state[strat:]
        #     temp1 = self.capacity-self.job_size[job_index]-np.array(tem1[::-1])
        #     # print(temp)
        #     bacth_index = np.where(temp1 > 0, temp1, 10000).argmin()



        done = bool(
            self.steps_beyond_done >= self.num_jobs-1
        )
        if done:
            pass
            #print(np.sum(self.buffer1))
            # self.state[strat+bacth_index] += self.job_size[job_index]
            # if self.state[strat+bacth_index] <= self.capacity:
            #     if self.state[bacth_index] < self.process_time[job_index]:
            #         reward = self.state[bacth_index]
            #         self.state[bacth_index] = self.process_time[job_index]
            #     else:
            #         reward = self.process_time[job_index]
            # else:
            #     reward = 0
            #print(np.sum(self.state[:100]))
        else:
            self.state[strat+bacth_index] += self.job_size[job_index]#eat job as reward
            if self.state[strat+bacth_index] <= self.capacity:#前一百个process_time,后一百个job_size
                if self.state[bacth_index] <= self.process_time[job_index]:
                    reward = self.state[bacth_index]
                    self.state[bacth_index] = self.process_time[job_index]
                else:
                    reward = self.process_time[job_index]
            else:
                reward = 0
            self.steps_beyond_done += 1

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = np.array([0.0]*self.num_states)
        self.buffer1 = np.array([])
        self.steps_beyond_done = 0
        return np.array(self.state, dtype=np.float32)
    def close(self):
        print("env close")


