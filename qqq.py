import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from random import choice
from sklearn import ensemble
from collections import  Counter

class MinEnv(gym.Env):


    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf

    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, action_list,num_machines, num_actions,hrr,power_pre,power_s,power_r,num_jobs,s_time,p_time,r_time,max_time,arrival_time,threshold1,threshold2,lower_time,lower_energy):
        self.num_machines= num_machines
        self.num_actions = num_actions
        self.max_state_value=np.array([1.0, 1.0, 5.0, 5.0])
        self.hrr=hrr
        self.power_pre=power_pre
        self.power_s = power_s
        self.power_r = power_r
        self.num_jobs = num_jobs
        self.action_list = action_list
        self.s_time = s_time # 工件熔炼时间
        #print(self.s_time)
        self.reward_parm = [10*np.random.random() for x in range(8)]  # 8个 reward 参数值
        self.p_time = p_time # 工件预热时间
        self.r_time = r_time  # 各工件精炼时间列表
        self.max_time=max_time
        self.arrival_time = arrival_time # 各工件到达时间列表
        self.completion_time = []  # 各工件完成时间列表
        self.idle_time=[]  #各工件后续的空闲时间列表
        self.comp_time =[]
        self.comp_energy=[]
        self.threshold1= threshold1
        self.threshold2 = threshold2 # 阈值λ2
        self.N1 = 0  # 工件之间的空闲时间小于阈值λ1的个数
        self.N2 = 0  # 工件之间的空闲时间(大于阈值λ1，小于阈值λ2)的个数
        self.completed_J = [] # 已经生产完成的工件的列表
        self.lower_time=lower_time
        self.lower_energy=lower_energy
        #self.process_time=np.array([10+50*np.random.random() for x in range(self.num_jobs)])
        #self.arrival_time =np.array([1 + 200* np.random.random() for x in range(self.num_jobs)])


        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Box(np.zeros(self.num_machines),self.max_state_value, dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = [0 for j in range(self.num_machines)]
        self.steps_beyond_done = 0
        self.temp_time = -2000
        self.temp_energy = 0.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cal_next_state(self,action2): # calculate next state
        x=[0]*self.num_machines
        if action2 == 0:
            gap_time = 0
        elif action2 == 1:
            gap_time = self.threshold1 * np.random.random()
        elif action2 == 2:
            gap_time = self.threshold1 + (self.threshold2 - self.threshold1) * np.random.random()
        else:
            gap_time = min(self.threshold2, max(self.arrival_time))

        if max(self.arrival_time[self.completed_J[-1]], self.temp_time+gap_time) - self.temp_time > self.threshold2:
            self.temp_time = max(self.arrival_time[self.completed_J[-1]], self.temp_time+gap_time) + \
                             self.p_time[self.completed_J[-1]] + self.r_time[self.completed_J[-1]] \
                             + self.s_time[self.completed_J[-1]]
            self.temp_energy += self.power_pre * self.p_time[self.completed_J[-1]] + self.power_r * self.r_time[self.completed_J[-1]] \
                                + self.power_s * self.s_time[self.completed_J[-1]]
            # print("0-1:", self.temp_time, self.temp_energy)
            x[0] = self.N1 / (self.num_jobs - 1)
            x[1] = self.N2 / (self.num_jobs - 1)
            if [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J]:
                temp_arrival_time = max(
                    [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J])
            else:
                temp_arrival_time = -1
            x[2] = (max(self.temp_time, temp_arrival_time) + sum([self.r_time[j] + self.hrr*self.s_time[j]
                                                                       for j in range(self.num_jobs)
                                                                       if j not in self.completed_J]) - self.lower_time) / self.lower_time
            x[3] = (self.temp_energy + sum(
                [self.power_r * self.r_time[j] +self.power_s  * self.s_time[j] + self.power_pre * self.p_time[j] for j in range(self.num_jobs)
                 if j not in self.completed_J]) - self.lower_energy) / self.lower_energy

        elif max(self.arrival_time[self.completed_J[-1]], self.temp_time+gap_time) - self.temp_time > self.threshold1:
            self.temp_time = max(self.arrival_time[self.completed_J[-1]], self.temp_time+gap_time) + \
                             self.r_time[self.completed_J[-1]] \
                             + self.s_time[self.completed_J[-1]]
            self.temp_energy += self.power_r * self.r_time[self.completed_J[-1]] \
                                + self.power_s * self.s_time[self.completed_J[-1]]
            x[0] = self.N1 / (self.num_jobs - 1)
            self.N2 = self.N2 + 1
            x[1] = self.N2 / (self.num_jobs - 1)
            if [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J]:
                temp_arrival_time = max(
                    [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J])
            else:
                temp_arrival_time = -1
            x[2] = (max(self.temp_time+gap_time, temp_arrival_time)
                    + sum([self.r_time[j] + self.hrr*self.s_time[j] for j in range(self.num_jobs) if
                           j not in self.completed_J]) - self.lower_time) / self.lower_time
            x[3] = (self.temp_energy + sum(
                [self.power_r * self.r_time[j] + self.power_s  * self.s_time[j] + self.power_pre * self.p_time[j] for j in range(self.num_jobs)
                 if j not in self.completed_J]) - self.lower_energy) / self.lower_energy
            # print("0-2:", self.temp_time, self.temp_energy)
        else:
            self.temp_time = max(self.arrival_time[self.completed_J[-1]], self.temp_time+gap_time) + \
                             +self.r_time[self.completed_J[-1]] \
                             + self.hrr * self.s_time[self.completed_J[-1]]
            self.temp_energy += self.power_r * self.r_time[self.completed_J[-1]] \
                                + self.hrr * self.power_s* self.s_time[self.completed_J[-1]]
            self.N1 = self.N1 + 1
            x[0] = self.N1 / (self.num_jobs - 1)
            x[1] = self.N2 / (self.num_jobs - 1)
            if [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J]:
                temp_arrival_time = max(
                    [self.arrival_time[j] for j in range(self.num_jobs) if j not in self.completed_J])
            else:
                temp_arrival_time = -1
            x[2] = (max(self.temp_time+gap_time, temp_arrival_time) + sum([self.r_time[j] + self.hrr * self.s_time[j]
                                                                       for j in range(self.num_jobs)
                                                                       if j not in self.completed_J]) - self.lower_time) / self.lower_time
            x[3] = (self.temp_energy + sum(
                [self.power_r * self.r_time[j] + self.power_s * self.s_time[j] + self.power_pre * self.p_time[j] for j in range(self.num_jobs)
                 if j not in self.completed_J]) - self.lower_energy) / self.lower_energy
        return x

    def step(self, action, reward_parm):
        #print(action)
        reward=0.0
        #temp_time=-2000.0
        #temp_energy=0.0
        #x= self.state[:]
        actionin = self.action_list[action]
        if actionin==0:
           action1=0
           action2=0
        if actionin==1:
           action1=0
           action2=1
        if actionin==2:
           action1=0
           action2=2
        if actionin==3:
           action1=0
           action2=3
        if actionin==4:
           action1=1
           action2=0
        if actionin==5:
           action1=1
           action2=1
        if actionin==6:
           action1=1
           action2=2
        if actionin==7:
           action1=1
           action2=3
        if actionin==8:
           action1=2
           action2=0
        if actionin==9:
           action1=2
           action2=1
        if actionin==10:
           action1=2
           action2=2
        if actionin==11:
           action1=2
           action2=3

        if action1 == 0:
            unschedule_job_arrival_time=[]
            for j in range(self.num_jobs):
                if j not in self.completed_J:
                    unschedule_job_arrival_time.append(self.arrival_time[j])
                else:
                    unschedule_job_arrival_time.append(self.arrival_time[j]+10000000)
            self.completed_J.append(unschedule_job_arrival_time.index(min(unschedule_job_arrival_time)))
            x=self.cal_next_state(action2)
                #print("0-3:", self.temp_time, self.temp_energy)
        elif action1 == 1:
            unschedule_job=[]
            for j in range(self.num_jobs):
                if j not in self.completed_J:
                    unschedule_job.append(j)
            self.completed_J.append(choice(unschedule_job))
            x=self.cal_next_state(0)

        elif action1 == 2:
            unschedule_job=[]
            arrival_semlt_time= []
            for j in range(self.num_jobs):
                if j not in self.completed_J:
                    unschedule_job.append(j)
                    arrival_semlt_time.append(self.hrr*self.s_time[j]+self.r_time[j])
            self.completed_J.append(unschedule_job[arrival_semlt_time.index(max(arrival_semlt_time))])
            x = self.cal_next_state(0)
        #temp_comp_before=max(self.state)
        #temp_comp_after=max(x)
        #self.state = x[:]
        #print(x)

        done = bool(
            self.steps_beyond_done >=self.num_jobs-1
        )

        if done:
            if x[2]<self.state[2]:
                reward += reward_parm[0]
            else:
                reward -= reward_parm[1]
            if x[3]<self.state[3]:
                reward += reward_parm[2]
            else:
                reward -= reward_parm[3]
            if x[0] >self.state[0]:
                reward += reward_parm[4]
            else:
                reward -= reward_parm[5]
            if x[1] > self.state[1]:
                reward += reward_parm[6]
            else:
                reward -= reward_parm[7]
            #print(self.temp_time,self.temp_energy)
           # filename = 'data.txt'  # 如果没有文件，自动创建
          #  with open(filename, 'a') as file_object:
           #     file_object.write(str(self.temp_time))
           #     file_object.write(" ")
            #    file_object.write(str(self.temp_energy))
            #    file_object.write("\n")
            self.comp_time.append(self.temp_time)
            self.comp_energy.append(self.temp_energy)
        else:
            self.steps_beyond_done += 1
            #print(reward_parm)
            if x[2] < self.state[2]:
                reward += reward_parm[0]
            else:
                reward -= reward_parm[1]
            if x[3] < self.state[3]:
                reward += reward_parm[2]
            else:
                reward -= reward_parm[3]
            if x[0] > self.state[0]:
                reward += reward_parm[4]
            else:
                reward -= reward_parm[5]
            if x[1] > self.state[1]:
                reward += reward_parm[6]
            else:
                reward -= reward_parm[7]
        self.state = x[:]
        #print(self.reward_parm)
        #print(self.completed_J)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = [0 for j in range(self.num_machines)]
        self.steps_beyond_done = 0
        self.temp_time = -20000
        self.temp_energy = 0.0
        self.completed_J = []
        self.N1 = 0
        self.N2 = 0
        return np.array(self.state, dtype=np.float32)

   # def render(self):



    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
