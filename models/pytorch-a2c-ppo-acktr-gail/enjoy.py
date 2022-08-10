import argparse
import json
import os
import sys

import typing as t

import numpy as np
import torch

from pathlib import Path
from collections import defaultdict

from a2c_ppo_acktr.envs import VecPyTorch, make_vec_envs
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize

import gym_linearnavigation

sys.path.append('a2c_ppo_acktr')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='LinearNavigationVisualOriginal-v2',
    help='gym environment to use')
parser.add_argument(
    '--model-path',
    help='path to the model .pt')
parser.add_argument(
    '--output-path',
    help='path of the json data output file')
parser.add_argument(
    '--n-episodes',
    default=20,
    type=int,
    help='number of episodes')
parser.add_argument(
    '--render',
    default=False,
    type=bool,
    help='if true, calls the render function on the environment at each timestep'
)
parser.add_argument(
    '--block-recurrent',
    type=str,
    help='stops the recurrent and readout activity of these neurons'
)
parser.add_argument(
    '--block-readout',
    type=str,
    help='stops these neurons from being used to read out actions'
)
parser.add_argument(
    '--recurrent-override-values',
    type=str,
    help='override recurrent values, e.g. None,0.3,0.4,None,0.4... (length must be equal to number of recurrent cells)'
)
parser.add_argument(
    '--readout-override-values',
    type=str,
    help='override readout values, e.g. None,0.3,0.4,None,0.4... (length must be equal to number of recurrent cells)'
)


args = parser.parse_args()


args.det = False

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False)

# Get a render function
render_func = get_render_func(env)

if 'STATE' not in Path(args.model_path).name:
    print(f'Converting .pt to STATE model type for backwards compatibility')

from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr import algo

pkl = torch.load(args.model_path, map_location='cpu')
if len(pkl) == 3:
    print(f'New generation model found')
    actor_critic_state, obs_rms, base_kwargs = pkl
elif len(pkl) == 2:
    print(f'Old generation model found - no base_kwargs available. Creating default settings.')
    actor_critic_pkl, obs_rms = pkl
    actor_critic_state = actor_critic_pkl.state_dict()
    base_kwargs = {'recurrent': True}
else:
    raise NotImplementedError

actor_critic = Policy(env.observation_space.shape,
                      env.action_space,
                      base_kwargs=base_kwargs)
actor_critic.load_state_dict(actor_critic_state)
Policy = algo.PPO(actor_critic, 0.5, 4, 4, 0.5, 0.1, lr=0.5, eps=0.5, max_grad_norm=0.5)  # Dummy variables since algo isn't important here

vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1,
                                      actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()


total_reward = 0

data = defaultdict(list)
activations = defaultdict(list)
dones = 0
trials = []
location = []
trial_type = []


def parse_mask(to_block_raw: t.Union[str, None]) -> t.Union[torch.Tensor, None]:
    if to_block_raw is not None:
        print(f'Parsing mask: {to_block_raw}')
        to_block = [int(x) for x in to_block_raw.split(',')]
        assert len(set(to_block)) == len(to_block)
        mask = torch.ones(actor_critic.dist.linear.in_features)
        for neuron in to_block:
            mask[neuron] = 0
        print(f'Parsed mask: {mask}')
        return mask
    else:
        return None


def parse_override(raw_override_arg: str) -> t.List[t.Union[None, float]]:
    if raw_override_arg is None:
        return

    override = []
    for value in raw_override_arg.split(','):
        if value == 'None':
            override.append(None)
        else:
            override.append(float(value))

    assert len(override) == actor_critic.recurrent_hidden_state_size

    return override


readout_mask = parse_mask(args.block_readout)
recurrent_mask = parse_mask(args.block_recurrent)
recurrent_override_values = parse_override(args.recurrent_override_values)
readout_override_values = parse_override(args.readout_override_values)


while dones < args.n_episodes:
    with torch.no_grad():
        value, action, _, recurrent_hidden_states = actor_critic.act(
            obs, recurrent_hidden_states, masks, deterministic=args.det,
            readout_mask=readout_mask, recurrent_mask=recurrent_mask, recurrent_override_values=recurrent_override_values, readout_override_values=readout_override_values)

    # Obser reward and next obs
    obs, reward, done, infos = env.step(action)
    info = infos[0]

    masks.fill_(0.0 if done else 1.0)

    total_reward += reward[0][0]

    if args.render:
        env.render()

    location.append(env.envs[0].env.location)

    data['location'].append(env.envs[0].env.location)

    for i, a in enumerate(recurrent_hidden_states.squeeze()):
        activations[i].append(float(a))

    if done:
        print(f'{args.model_path} - {dones} - Reward: {total_reward}')
        total_reward = 0

        dones += 1

    data['trial'].append(dones)

    trials.append(dones)


# Ugly way to convert from int64 to float to make it serialisable
float_locations = [float(f) for f in data['location']]
data['location'] = float_locations
data['rates'] = activations  # Raw unit rates

with open(args.output_path, 'w') as path:
    json.dump(data,  path)