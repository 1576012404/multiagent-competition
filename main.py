from policy import LSTMPolicy, MlpPolicyValue
import gym
import gym_compete
import pickle
import sys
import argparse
import tensorflow as tf
import numpy as np
from collections import deque

def load_from_file(param_pkl_path):
    with open(param_pkl_path, 'rb') as f:
        params = pickle.load(f)
    return params

def setFromFlat(var_list, flat_params):
    print("var_list",var_list)
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    tf.get_default_session().run(op, {theta: flat_params})

def run(config):
    if config.env == "kick-and-defend":
        env = gym.make("kick-and-defend-v0")
        policy_type = "lstm"
    elif config.env == "run-to-goal-humans":
        env = gym.make("run-to-goal-humans-v0")
        policy_type = "mlp"
    elif config.env == "run-to-goal-ants":
        env = gym.make("run-to-goal-ants-v0")
        policy_type = "mlp"
    elif config.env == "you-shall-not-pass":
        env = gym.make("you-shall-not-pass-humans-v0")
        policy_type = "mlp"
    elif config.env == "sumo-humans":
        env = gym.make("sumo-humans-v0")
        policy_type = "lstm"
    elif config.env == "sumo-ants":
        env = gym.make("sumo-ants-v0")
        policy_type = "lstm"
    else:
        print("unsupported environment")
        print("choose from: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
        sys.exit()

    param_paths = config.param_paths
    print("param_paths",param_paths)

    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)

    sess = tf.Session(config=tf_config)
    sess.__enter__()

    policy = []
    for i in range(2):
        scope = "policy" + str(i)
        if policy_type == "lstm":
            policy.append(LSTMPolicy(scope=scope, reuse=False,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hiddens=[128, 128], normalize=True))
        else:
            policy.append(MlpPolicyValue(scope=scope, reuse=False,
                                         ob_space=env.observation_space.spaces[i],
                                         ac_space=env.action_space.spaces[i],
                                         hiddens=[64, 64], normalize=True))

    # initialize uninitialized variables
    sess.run(tf.variables_initializer(tf.global_variables()))


    params = [load_from_file(param_pkl_path=path) for path in param_paths]
    for i in range(len(policy)):
        setFromFlat(policy[i].get_variables(), params[i])


    max_episodes = config.max_episodes
    num_episodes = 0
    total_reward = [0.0  for _ in range(len(policy))]
    total_scores = [0 for _ in range(len(policy))]
    # total_scores = np.asarray(total_scores)
    observation = env.reset()
    print("-"*5 + " Episode %d " % (num_episodes+1) + "-"*5)
    iFrame=0
    iReward = np.zeros((2,))
    reward_list = deque(maxlen=100)
    reward_list2 = deque(maxlen=100)
    win_times = [0, 0]

    while num_episodes < max_episodes:
        env.render()
        action=[policy[i].act(stochastic=True, observation=observation[i])[0]
                        for i in range(len(policy))]
        # action[1]=np.zeros((8),dtype=np.float32)
        # action = tuple(action)
        observation, reward, done, infos = env.step(action)
        iReward += reward
        iFrame+=1
        # print("obs",len(observation),observation[0].shape,len(done),done,len(reward),reward)
        # for i in range(len(policy)):
        #     total_reward[i] += reward[i]

        if done[0]:

            iWin = -1
            if 'winner' in infos[0]:
                iWin = 0
                win_times[0]+=1
            elif 'winner' in infos[1]:
                iWin = 1
                win_times[1] += 1

            reward_list.append(iReward[0])
            reward_list2.append(iReward[1])
            print("winer:",iWin,"%s/%s"%(win_times[0],win_times[1]), "frames:",iFrame, "agent0", iReward[0], sum(reward_list) / len(reward_list), "agnet2",
                  iReward[1], sum(reward_list2) / len(reward_list2))

            num_episodes += 1
            # draw = True
            # for i in range(len(policy)):
            #     if 'winner' in infos[i]:
            #         draw = False
            #         total_scores[i] += 1
            #         print("Winner: Agent {}, Scores: {}, Total Episodes: {},Step:{}".format(i, total_scores, num_episodes,iFrame))
            # if draw:
            #     print("Game Tied: Agent {}, Scores: {}, Total Episodes: {},Step:{}".format(i, total_scores, num_episodes,iFrame))
            observation = env.reset()
            iFrame = 0
            iReward = np.zeros((2,))
            # total_reward = [0.0  for _ in range(len(policy))]
            for i in range(len(policy)):
                policy[i].reset()
            if num_episodes < max_episodes:
                print("-"*5 + "Episode %d" % (num_episodes+1) + "-"*5)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Environments for Multi-agent competition")
    p.add_argument("--env", default="sumo-humans", type=str, help="competitive environment: run-to-goal-humans, run-to-goal-ants, you-shall-not-pass, sumo-humans, sumo-ants, kick-and-defend")
    p.add_argument("--param-paths", nargs='+', required=True, type=str)
    p.add_argument("--max-episodes", default=10, help="max number of matches", type=int)

    config = p.parse_args()
    run(config)
