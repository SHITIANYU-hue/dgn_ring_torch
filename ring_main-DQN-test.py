# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
import pandas as pd
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment
from ring_Env import para_produce_rl, Experiment
import logging

import datetime
import numpy as np
import time
import os
from DQN import DQN
from bufferdqn import ReplayBuffer
from flow.core.params import SumoParams
# define some parameters
import pandas as pd
import os
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F


from config import *


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')


# define some environment parameters
exp_tag = "idm_ring"
mkdir('{}_results'.format(exp_tag))
agent_num =10
train_test = 1  # define train(1) or test(2)
num_runs = 1
# build up settings
flow_params = para_produce_rl(NUM_AUTOMATED=agent_num)
env = Experiment(flow_params=flow_params).env
rl_actions = None
convert_to_csv = True
model_path = "./model/{0}_model.ckpt".format(exp_tag)
sim_params = SumoParams(sim_step=0.1, render=False)
num_steps = 3600


n_ant = agent_num
observation_space = 3
n_actions = 1

buff = ReplayBuffer(capacity)
model = DQN(n_ant, observation_space, hidden_dim, n_actions)
model_tar = DQN(n_ant, observation_space, hidden_dim, n_actions)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

O = np.ones((batch_size, n_ant, observation_space))
Next_O = np.ones((batch_size, n_ant, observation_space))


save_interal = 5
rets = []
mean_rets = []
ret_lists = []
vels = []
mean_vels = []
std_vels = []
outflows = []
t = time.time()
times = []
vehicle_times = []
ploss = 0
qloss = 0
reg_loss = 0
results = []
scores = []
losses = []
# save simulation videos


def render(render_mode='sumo_gui'):
    from flow.core.params import SimParams as sim_params
    sim_params.render = True
    save_render = True
    setattr(sim_params, 'num_clients', 1)
    # pick your rendering mode
    if render_mode == 'sumo_web3d':
        sim_params.num_clients = 2
        sim_params.render = False
    elif render_mode == 'drgb':
        sim_params.render = 'drgb'
        sim_params.pxpm = 4
    elif render_mode == 'sumo_gui':
        sim_params.render = False  # will be set to True below
    elif render_mode == 'no_render':
        sim_params.render = False
    if save_render:
        if render_mode != 'sumo_gui':
            sim_params.render = 'drgb'
            sim_params.pxpm = 4
        sim_params.save_render = True



def calculate_info(env):
    # calculate the car flow
    aver_speed = 0
    aver_hdw=0
    
    for veh_id in env.k.vehicle.get_ids():
        aver_speed += env.k.vehicle.get_speed(veh_id)

    for veh_id in env.k.vehicle.get_ids():
        aver_hdw += env.k.vehicle.get_headway(veh_id)

    aver_speed /= len(env.k.vehicle.get_ids())
    aver_hdw /= len(env.k.vehicle.get_ids())


    speeds_by_lane = {}

    # loop over all vehicle IDs
    for veh_id in env.k.vehicle.get_ids():
        # get the speed and lane ID for this vehicle
        speed = env.k.vehicle.get_speed(veh_id)
        lane_id = env.k.vehicle.get_lane(veh_id)

        # if this lane has not been seen before, add it to the dictionary
        if lane_id not in speeds_by_lane:
            speeds_by_lane[lane_id] = []

        # add the speed of this vehicle to the list for its lane
        speeds_by_lane[lane_id].append(speed)

    # loop over each lane and compute the average speed
    average_speeds_bylane = {}
    for lane_id in speeds_by_lane:
        avg_speed = sum(speeds_by_lane[lane_id]) / len(speeds_by_lane[lane_id])
        average_speeds_bylane[lane_id] = avg_speed



    return aver_speed, aver_hdw/env.k.network.length(), average_speeds_bylane


def calculate_info_(env):
    # calculate the car flow

    data={}
    for veh_id in env.k.vehicle.get_ids():
        hdw  = env.k.vehicle.get_headway(veh_id)
        speed = env.k.vehicle.get_speed(veh_id)
        position = env.k.vehicle.get_position(veh_id)
        lane_id = env.k.vehicle.get_lane(veh_id)
        data_={veh_id:[hdw,speed,position,lane_id]}
        data.update(data_)

    return data



model.load_state_dict(torch.load('/home/changquan/dgn_ring_torch/save_model/model_95'))
model.eval()
for i_episode in range(num_runs):
    # logging.info("Iter #" + str(i))
    print('episode is:', i_episode)
    ret = 0
    ret_list = []
    obs = env.reset()

    aset = []
    speed_by_lanes=[]
    score = 0
    for j in range(num_steps):
        # manager actions
        # convert state into values
        state_ = np.array(list(obs.values())).reshape(agent_num, -1).tolist()


        state_ = torch.tensor(np.asarray([state_]), dtype=torch.float)

        q = model(state_)[0]
        for i in range(n_ant):
            if np.random.rand() < epsilon:
                a = np.random.randint(n_actions)
            else:
                a = q[i].argmax().item()
            aset.append(a)

        action_dict = {}
        k = 0
        for key, value in obs.items():
            action_dict[key] = aset[k]+ 0.5*np.random.uniform(-1,0,1).item()
            k += 1

        speed_limit = 20
        #print('action_dict:',action_dict)
        #next_state, reward, done, _ = env.step(action_dict, speed_limit)
        next_state, reward, done, _ = env.step(action_dict)

        next_state_ = np.array(list(next_state.values())
                               ).reshape(agent_num, -1).tolist()
        done_ = np.array(list(done.values())).reshape(1, -1).tolist()

        for i in range(len(done_)):
            if done_ != False:
                done_ = 1
                break

        reward_ = np.array(list(reward.values())).reshape(1, -1).tolist()
        # print('reward',np.average(reward_))
        buff.add(np.array(state_), aset, np.average(reward_), np.array(
            next_state_), done_)
        obs = next_state
        # print('reward',reward)

        score += sum(list(reward.values()))
        avg_speed,avg_hw,speed_by_lane =calculate_info(env)
        speed_by_lanes.append(speed_by_lane)
        print('score/speed_by_lane',speed_by_lane)
        np.save(f'score/speed_by_lanes_{exp_tag}.npy',speed_by_lanes)
    scores.append(score/num_steps)
    print('scores',scores)


env.terminate()
