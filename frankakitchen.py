#Author: Damian Grunert
#Date: 27-02-2024
#Content: An implementation of a project using my ODT implementation

import gym
import d4rl
import torch
import numpy as np
import ODT
import pickle

env = gym.make('hopper-medium-v2')

x = env.reset()
env.step(env.action_space.sample())

#Wrapper of the franka kitchen environment to interact with using pytorch
class kitchenEnv:
    def __init__(self, env) -> None:
        self.env = env

    def reset(self):
        return self.env.reset().astype('float32')
    
    def step(self, action):
        state, reward, done, _ = env.step(action.numpy().reshape(action.shape[1]))
        return state.astype('float32'), reward, done, _
    
    def render(self, args = None):
        return self.env.render(args)
    

#Contains a dataset of N observations 
dataset = env.get_dataset()
observations = dataset['observations']
actions = dataset['actions']
rewards = dataset['rewards'].reshape(-1, 1)
terminals = dataset['terminals']

def split_array(arr, bool_arr):
    splits = np.where(bool_arr)[0]  # Find indices where boolean array is True
    split_arrays = []
    start_idx = 0
    for end_idx in splits:
        split_arrays.append(torch.tensor(arr[start_idx:end_idx + 1], dtype=torch.float32))
        start_idx = end_idx + 1
    # Append the last split
    if start_idx < len(arr):
        split_arrays.append(torch.tensor(arr[start_idx:], dtype=torch.float32))
    return split_arrays

observation_list = split_array(observations, terminals)
action_list = split_array(actions, terminals)
reward_list = split_array(rewards, terminals)

print('Creating model')

OnlineDecisionTransformer = ODT.ODT(observations.shape[1], actions.shape[1], 16, 8, 1024, 3, 200, 800, 1, [-1, 1])
OnlineDecisionTransformer.add_data(reward_list, observation_list, action_list)

print('Data loaded. Starting training.')

corr_opt = torch.optim.Adam(OnlineDecisionTransformer.transformer.parameters())
entr_opt = torch.optim.Adam([OnlineDecisionTransformer.lamda_ln])

torch.cuda.empty_cache()

for i in range(600):
    print(i, OnlineDecisionTransformer.train(50, 138, corr_opt, entr_opt, norm_clip=0.25))
    if i % 10 == 0:
        print(i, OnlineDecisionTransformer.render_episode(8400, kitchenEnv(env), max_episode_len=1000, iterations=1, name = str(i), path = "./visualization_kitchen/"))
        with open('./variants/transformer_'+str(i), 'wb') as outp:
            pickle.dump(OnlineDecisionTransformer, outp, pickle.HIGHEST_PROTOCOL)

print("Finished")