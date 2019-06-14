from gym import Env as gymEnv
from gym import spaces
from baseline.ppo2 import ppo2
from baseline.common.vec_env.subproc_vec_env_mul import SubprocVecEnvMulti

# from my_dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

from baseline.common.vec_env.vec_monitor import VecMonitorMulti
from baseline.common.vec_env.vec_normalize import VecNormalizeMulti
from baseline.common.vec_env import VecEnvWrapper

import gym
import gym_compete
from baseline import logger
import datetime
import time
from collections import deque

import tensorflow as tf
import numpy as np
from itertools import chain

class Agent(object):

    def __init__(self,agent_id,observation_space,action_space):
        self.id = agent_id
        self.observation_space=observation_space
        self.action_space=action_space

class Env(gymEnv):
    spec = None
    def __init__(self,iIndex,team_size=1):
        super(gymEnv,self).__init__()
        observation_space=spaces.Box(-1, 1, (122,))
        action_space=spaces.Box(-1, 1, (8,))
        self.observation_space=[spaces.Box(-1, 1, (122,))for i in range(2*team_size)]
        self.action_space=[spaces.Box(-1, 1, (8,))for i in range(2*team_size)]
        self.agents = [Agent(i, observation_space, action_space) for i in range(2 * team_size)]
        self.env= gym.make("run-to-goal-ants-v0")
        self.n_agent=2*team_size
        self.m_Index=iIndex
        self.m_MaxFrame=500
        self.n_step=0
        self.m_Eposide0=deque(maxlen=100)
        self.m_Eposide1 = deque(maxlen=100)
        self.m_EpCount=0

    def reset(self):
        obs = self.env.reset()
        self.n_step = 0
        return obs

    def step(self,actions):
        self.n_step+=1
        observation, reward, done, infos = self.env.step(actions)
        if self.n_step>=self.m_MaxFrame:
            done=[True for _ in range(self.n_agent)]
        if done[0] and self.m_Index==0:
            # if 'winner' in infos[0]:
            #     print("0 win")
            # if 'winner' in infos[1]:
            #     print("1 win")
            self.m_Eposide0.append('winner' in infos[0])
            self.m_Eposide1.append('winner' in infos[1])
            self.m_EpCount+=1
            if self.m_EpCount%10==0:
                iLen=len(self.m_Eposide0)
                print("winrate",sum(self.m_Eposide0)/iLen,sum(self.m_Eposide1)/iLen,iLen)
        if self.m_Index==0:
            self.render()
        return observation,reward,done,infos

    def render(self,mode="Human"):
        self.env.render()




def Eval():
    def EnvFunc(iIndex):
        def InnerFunc():
            oEnv=Env(iIndex)
            return oEnv
        return InnerFunc

    def linear_schedule(initial_value):
        def func(process):
            return process * initial_value
        return func

    learning_rate = linear_schedule(3e-4)
    clip_range = linear_schedule(0.2)
    n_timesteps = int(0)
    hyperparmas = {'nsteps': 1024, 'noptepochs': 10, 'nminibatches': 32, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.00}

    num_env = 1
    num_agent=2
    env = SubprocVecEnvMulti([EnvFunc(i) for i in range(num_env)],num_agent=num_agent)
    env=VecMonitorMulti(env)
    env=VecNormalizeMulti(env)

    act_list = ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=n_timesteps,
        log_interval=4,
        save_interval=100,
        load_path=["baselineLog/ppobaselines-2019-06-12-14-47-25-043390/checkpoints/0-03400",
                   "baselineLog/ppobaselines-2019-06-12-14-47-25-043390/checkpoints/1-03400"],

        **hyperparmas,

        value_network="copy"
    )

    obs = env.reset()
    print("obs", obs[0].shape)
    bDone = False
    iFrame = 0
    iReward = np.zeros((2,))
    reward_list = deque(maxlen=100)
    reward_list2 = deque(maxlen=100)
    while not bDone:
        all_agent_action=[]
        for i in range(num_agent):
            actions = act_list[i].step(obs[i])[0]
            all_agent_action.append(actions)
        # print("all_agent_action",all_agent_action)

        all_env_action=list(zip(*all_agent_action))
        # print("all_env_action",all_env_action)
        obs, reward, done, info = env.step(all_env_action)
        iReward += np.array([reward[0][0],reward[1][0]])
        # time.sleep(0.01)
        # print("reward",reward)
        iFrame += 1
        # env.render()
        if done[0][0]:
            iWin = -1
            if 'winner' in info[0][0]:
                iWin = 0
            elif 'winner' in info[1][0]:
                iWin = 1

            reward_list.append('winner' in info[0][0])
            reward_list2.append('winner' in info[1][0])
            print("winer:", iWin, iFrame, "agent0", iReward[0], sum(reward_list) / len(reward_list), "agnet2",
                  iReward[1], sum(reward_list2) / len(reward_list2))

            iFrame = 0
            iReward = np.zeros((2,))
            obs = env.reset()




if __name__=="__main__":
    # RomdomPlay()
    Eval()
