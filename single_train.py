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


class Env(gymEnv):
    spec = None
    def __init__(self,):
        super(gymEnv,self).__init__()
        self.action_space=spaces.Box(-1,1,(8,))
        self.observation_space = spaces.Box(-1, 1, (122,))
        self.env= gym.make("run-to-goal-ants-v0")
        self.m_MaxFrame=400
        self.n_step=0



    def reset(self):
        self.n_step=0
        obs = self.env.reset()
        return obs[0]

    def step(self,action):
        self.n_step+=1
        actions=(action,np.zeros((8,),np.float32))
        observation, reward, done, infos = self.env.step(actions)
        # print("obs",len(observation),observation[0].shape,reward,done)
        bDone=done[0]
        if self.n_step>=self.m_MaxFrame:
            bDone=True

        return observation[0],reward[0],bDone,{}

    def render(self,mode):
        self.env.render()


def RomdomPlay():
    oEnv=Env()
    bDone=False
    iFrame=0
    while not bDone:
        action = np.random.uniform(-1, 1, size=(3,))
        _obs,reward,done,_=oEnv.step(action)
        iFrame+=1
        oEnv.render()
        if done:
            _obs=oEnv.reset()
            print("reset.................")
            iFrame=0

def Train():
    logdir = "baselineLog/ppo" + datetime.datetime.now().strftime("baseliens-%Y-%m-%d-%H-%M-%S-%f")
    logger.configure(logdir, ["tensorboard", "stdout"])


    def EnvFunc(iSeed):
        def InnerFunc():
            oEnv=Env()
            return oEnv
        return InnerFunc

    def linear_schedule(initial_value):
        def func(process):
            return process * initial_value
        return func

    learning_rate = linear_schedule(3e-4)
    clip_range = linear_schedule(0.2)
    n_timesteps = int(1e8)
    hyperparmas = {'nsteps': 1024, 'noptepochs': 10, 'nminibatches': 32, 'lr': learning_rate, 'cliprange': clip_range,
                   'vf_coef': 0.5, 'ent_coef': 0.0}


    num_env = 28
    env = SubprocVecEnv([EnvFunc(i) for i in range(num_env)])
    env=VecMonitor(env)
    env=VecNormalize(env)

    act = ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=n_timesteps,
        save_interval=100,
        log_interval=4,
        # load_path="/tmp/openai-2019-05-30-11-53-14-660522/checkpoints/16000",
        **hyperparmas,

        value_network="copy"
    )


if __name__=="__main__":
    # RomdomPlay()
    Train()






