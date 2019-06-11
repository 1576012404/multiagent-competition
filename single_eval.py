from gym import Env as gymEnv
from gym import spaces
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.common.vec_env.vec_normalize import VecNormalize

import gym
import gym_compete
from baselines import logger
import datetime
import tensorflow as tf
import numpy as np
import time
from collections import deque

class Env(gymEnv):
    spec = None
    def __init__(self,):
        super(gymEnv,self).__init__()
        self.action_space=spaces.Box(-1,1,(8,))
        self.observation_space = spaces.Box(-1, 1, (122,))
        self.env= gym.make("run-to-goal-ants-v0")
        self.m_MaxFrame=500
        self.n_step=0



    def reset(self):
        self.n_step=0
        obs = self.env.reset()
        return obs[0]

    def step(self,action):
        self.n_step+=1
        actions=(action,np.zeros((8,),np.float32))
        observation, reward, done, infos = self.env.step(actions)
        self.render()
        # print("obs",len(observation),observation[0].shape,reward,done)
        bDone=done[0]
        if self.n_step>=self.m_MaxFrame:
            bDone=True

        return observation[0],reward[0],bDone,infos[0]

    def render(self,mode="human"):
        obs=self.env.render()


def Eval():



    def EnvFunc(iSeed):
        def InnerFunc():
            oEnv=Env()
            return oEnv
        return InnerFunc

    def linear_schedule(initial_value):
        def func(process):
            return process * initial_value
        return func

    learning_rate = linear_schedule(5e-4)
    clip_range = linear_schedule(0.2)
    n_timesteps = int(0)
    hyperparmas = {'nsteps': 256, 'noptepochs': 8, 'nminibatches': 4, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.01}


    num_env = 1
    env = SubprocVecEnv([EnvFunc(i) for i in range(num_env)])
    env = VecNormalize(env,ob=True,ret=False)
    env=VecMonitor(env)

    act = ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=n_timesteps,
        save_interval=100,
        load_path="baselineLog/ppobaseliens-2019-06-05-23-16-45-061477/checkpoints/01300",
        **hyperparmas,
        value_network="copy"
    )
    print("???????????????")


    obs = env.reset()
    print("obs", obs.shape)
    bDone = False
    iFrame = 0
    iReward = 0
    iEposide=0
    iWin=0
    reward_list=deque(maxlen=100)
    while not bDone:
        action = act.step(obs)[0]
        obs, reward, done, infos = env.step(action)
        iReward += reward[0]
        # time.sleep(0.01)
        # print("reward",reward)
        iFrame += 1
        # env.render()
        if done[0]:
            iEposide+=1
            if "winner"in infos[0]:
                iWin+=1
            obs = env.reset()
            reward_list.append(iReward)
            print("win","winner"in infos[0], iWin/iEposide,iFrame, iReward,sum(reward_list)/len(reward_list))

            iFrame = 0
            iReward = 0

    # oEnv=Env()
    # obs=oEnv.reset()
    # print("obs",obs.shape)
    # oEnv.render()
    # bDone=False
    # iFrame=0
    # iReward=0
    # while not bDone:
    #     action = act.step(obs[None,:])[0]
    #     action=action[0]
    #     # print("action",action)
    #     # action=np.zeros((8,))
    #     # action[0:8]=1
    #     obs,reward,done,_=oEnv.step(action)
    #     iReward+=reward
    #     # time.sleep(0.01)
    #     # print("reward",reward)
    #     iFrame+=1
    #     oEnv.render()
    #     if done:
    #         obs=oEnv.reset()
    #         print("done.................",iFrame,iReward)
    #         iFrame=0
    #         iReward=0




if __name__=="__main__":
    # RomdomPlay()
    Eval()






