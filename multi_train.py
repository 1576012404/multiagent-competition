from gym import Env as gymEnv
from gym import spaces
from baselines.ppo2 import ppo2
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

# from my_dummy_vec_env import DummyVecEnv
# from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

# from baselines.common.vec_env.vec_monitor import VecMonitor
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env import VecEnvWrapper

import gym
import gym_compete
from baselines import logger
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
    def __init__(self,team_size=1):
        super(gymEnv,self).__init__()
        self.observation_space=observation_space= spaces.Box(-1, 1, (123,))
        self.action_space=action_space = spaces.Box(-1, 1, (8,))
        self.agents = [Agent(i, observation_space, action_space) for i in range(2 * team_size)]
        self.env= gym.make("run-to-goal-ants-v0")
        self.m_MaxFrame=400
        self.n_step=0

    def reset(self):
        self.n_step=0
        obs = self.env.reset()
        return obs

    def step(self,actions):
        self.n_step+=1
        observation, reward, done, infos = self.env.step(actions)
        bDone=np.any(done)
        # print("obs",len(observation),observation[0].shape,reward,done)
        if self.n_step>=self.m_MaxFrame:
            bDone=True
            # done=(np.ones(1,np.bool),np.ones(1,np.bool))
        return observation,reward,bDone,infos

    def render(self,mode="Human"):
        self.env.render()


class Self_Play_Wrapper(VecEnvWrapper):
    def __init__(self, venv,num_agent=2 ):
        VecEnvWrapper.__init__(self, venv,)
        self.real_num_envs=venv.num_envs
        self.num_agent=num_agent
        self.num_envs = venv.num_envs*num_agent

    def step_async(self, actions):
        actions=np.reshape(actions,(self.real_num_envs,self.num_agent,*self.action_space.shape))
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        # print("obs",obs.shape)
        obs = np.reshape(obs, (self.num_envs, *self.observation_space.shape))
        rews=np.reshape(rews,(self.num_envs,))
        dones=np.tile(dones,self.num_agent)

        # news = np.reshape(news, (self.num_envs,))

        infos=list(chain(*infos))
        return obs, rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        obs=np.reshape(obs,(self.num_envs,*self.observation_space.shape))
        return obs



class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, filename=None, keep_buf=0):
        VecEnvWrapper.__init__(self, venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.keep_buf = keep_buf
        self.num_agent=self.venv.num_agent
        if self.keep_buf:
            self.epret_buf = deque([], maxlen=keep_buf)
            self.eplen_buf = deque([], maxlen=keep_buf)

    def reset(self):
        obs = self.venv.reset()
        self.eprets = np.zeros(self.num_envs, 'f')
        self.eplens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.eprets += rews
        self.eplens += 1
        newinfos = []
        for (i, (done, ret, eplen, info)) in enumerate(zip(dones, self.eprets, self.eplens, infos)):
            agent_id=i%self.num_agent
            # if agent_id!=0:
            #     continue
            info = info.copy()
            if done:
                epinfo = {'r': ret, 'l': eplen, 't': round(time.time() - self.tstart, 6)}
                info['episode%s'%agent_id] = epinfo
                # print("done info",info)
                if self.keep_buf:
                    self.epret_buf.append(ret)
                    self.eplen_buf.append(eplen)
                self.epcount += 1
                self.eprets[i] = 0
                self.eplens[i] = 0
            newinfos.append(info)

        return obs, rews, dones, newinfos






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
                   'vf_coef': 0.5, 'ent_coef': 0.00}

    num_env = 30
    env = SubprocVecEnv([EnvFunc(i) for i in range(num_env)])
    env=Self_Play_Wrapper(env)
    env=VecMonitor(env)
    env=VecNormalize(env)

    act = ppo2.learn(
        network="mlp",
        env=env,
        total_timesteps=n_timesteps,
        log_interval=4,
        save_interval=100,
        # load_path="baselineLog/ppobaseliens-2019-06-05-11-26-10-382460/checkpoints/00300",
        **hyperparmas,

        value_network="copy"
    )


if __name__=="__main__":
    # RomdomPlay()
    Train()





def RomdomPlay():
    oEnv=Env()
    bDone=False
    iFrame=0
    obs=oEnv.reset()
    print("reset",obs)
    while not bDone:
        action = np.random.uniform(-1, 1, size=(3,))
        _obs,reward,done,_=oEnv.step(action)
        print("step",reward,done,type(done),type(done[0]))
        iFrame+=1
        oEnv.render()
        if done[0]:
            _obs=oEnv.reset()
            print("reset.................",iFrame,len(_obs),_obs[0].shape)
            iFrame=0
