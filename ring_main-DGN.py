# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
import pandas as pd
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment
from DGN_Env import para_produce_rl, Experiment
import logging

import datetime
import numpy as np
import time
import os
from DGN import DGN
from buffer import ReplayBuffer
from flow.core.params import SumoParams
### define some parameters
import pandas as pd
import os
import torch.optim as optim

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from ES_VSL import ES_VSL, SGD
import multiprocessing as mp


from config import *
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')


## define some environment parameters
exp_tag="dgn_ring"
build_adj=2
mkdir('{}_results'.format(exp_tag))
agent_num=3
neighbors=3
train_test=1 ##define train(1) or test(2)
num_runs=100
## build up settings
flow_params = para_produce_rl(NUM_AUTOMATED=agent_num)
env = Experiment(flow_params=flow_params).env
rl_actions=None
convert_to_csv=True
model_path="./model/{0}_model.ckpt".format(exp_tag)
env.sim_params.emission_path='./{}_emission/'.format(exp_tag)
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='./{0}_emission/'.format(exp_tag))
num_steps = env.env_params.horizon



n_ant = agent_num
observation_space = 3
n_actions = 1


buff = ReplayBuffer(capacity)
model = DGN(n_ant,observation_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,observation_space,hidden_dim,n_actions)
model = model
model_tar = model_tar
optimizer = optim.Adam(model.parameters(), lr = 0.0001)

O = np.ones((batch_size,n_ant,observation_space))
Next_O = np.ones((batch_size,n_ant,observation_space))
Matrix = np.ones((batch_size,n_ant,n_ant))
Next_Matrix = np.ones((batch_size,n_ant,n_ant))

save_interal=20
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
ploss=0
qloss=0
reg_loss=0
results=[]
scores=[]
losses=[]
## save simulation videos
def render(render_mode='sumo_gui'):
    from flow.core.params import SimParams as sim_params
    sim_params.render=True
    save_render=True
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
def average(data):
    return sum(data)/len(data)

## todo how to define agent's relationship
if build_adj==1:
    # method 1:sort for the nearest speed vehicle
    def Adjacency( env ,neighbors=2):
        adj = []
        vels=np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
        orders = np.argsort(vels)
        for rl_id1 in env.k.vehicle.get_rl_ids():
            l = np.zeros([neighbors,len(env.k.vehicle.get_rl_ids())])
            j=0
            for k in range(neighbors):
                # modify this condition to define the adjacency matrix
                l[k,orders[k]]=1

            adj.append(l)
        return adj

if build_adj==2:
    # method2: sort for the nearest position vehicle
    def Adjacency(env ,neighbors=2):
        adj = []
        x_pos = np.array([env.k.vehicle.get_x_by_id(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
        headways = np.zeros([len(env.k.vehicle.get_rl_ids()),len(env.k.vehicle.get_rl_ids())])
        for d in range(len(env.k.vehicle.get_rl_ids())):
            headways[d,:] = abs(x_pos-x_pos[d])
        
        orders = np.argsort(headways)
        for rl_id1 in env.k.vehicle.get_rl_ids():
            l = np.zeros([neighbors,len(env.k.vehicle.get_rl_ids())])
            j=0
            for k in range(neighbors):
                # modify this condition to define the adjacency matrix
                l[k,orders[k]]=1

            adj.append(l)
        return adj

if build_adj==3:
    ## method 3: consider both speed and position
    def Adjacency(env ,neighbors=2):
        des_vel=5
        adj = []
        x_pos = np.array([env.k.vehicle.get_x_by_id(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
        x_vel = np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_rl_ids() ])
        headways = np.zeros([len(env.k.vehicle.get_rl_ids()),len(env.k.vehicle.get_rl_ids())])
        for d in range(len(env.k.vehicle.get_rl_ids())):
            headways[d,:] = abs(x_pos-x_pos[d])+x_vel/(des_vel*abs(x_vel-x_vel[d])+0.01)

        orders = np.argsort(headways)
        for rl_id1 in env.k.vehicle.get_rl_ids():
            l = np.zeros([neighbors,len(env.k.vehicle.get_rl_ids())])
            j=0
            for k in range(neighbors):
                # modify this condition to define the adjacency matrix
                l[k,orders[k]]=1

            adj.append(l)
        return adj



def calculate_aver_speed(env):
    # calculate the car flow
    aver_speed = 0
    for veh_id in env.k.vehicle.get_ids():
        aver_speed += env.k.vehicle.get_speed(veh_id)
    
    aver_speed /= len(env.k.vehicle.get_ids())
    print("aver_speed : ",aver_speed)
    return aver_speed

for i_episode in range(num_runs):
    # logging.info("Iter #" + str(i))
    print('episode is:',i_episode)
    ret = 0
    ret_list = []
    obs = env.reset()

    aset = []
    vec = np.zeros((1, neighbors))
    vec[0][0] = 1
    score=0 



    for j in range(num_steps):
        # manager actions
        # convert state into values
        state_ = np.array(list(obs.values())).reshape(agent_num,-1).tolist()

        adj = Adjacency(env ,neighbors=neighbors)

        state_= torch.tensor(np.asarray([state_]),dtype=torch.float) 
        adj_= torch.tensor(np.asarray(adj),dtype=torch.float)
        
        q = model(state_, adj_)[0]
        for i in range(n_ant):
            if np.random.rand() > epsilon:
                a = np.random.randint(n_actions)
            else:
                a = q[i].argmax().item()
            aset.append(a)

        action_dict = {}
        k=0
        for key,value in obs.items():
            action_dict[key]=aset[k]
            k+=1

        speed_limit = 20
        next_state, reward, done, _ = env.step(action_dict, speed_limit)

        next_adj = Adjacency(env ,neighbors=neighbors)


        next_state_ = np.array(list(next_state.values())).reshape(agent_num,-1).tolist()
        done_=np.array(list(done.values())).reshape(1,-1).tolist()

        for i in range(len(done_)):
            if done_!=False:
                done_=1
                break


        reward_ = np.array(list(reward.values())).reshape(1,-1).tolist()
        # print('reward',np.average(reward_))
        buff.add(np.array(state_),aset,np.average(reward_),np.array(next_state_),np.array(adj[-1]),np.array(next_adj[-1]), done_)
        obs = next_state
        # print('reward',reward)
        
        score += sum(list(reward.values()))
        
    aver_speed = calculate_car_flow(env)
    
    scores.append(score/num_steps)


    np.save('scores.npy',scores)

         ## calculate individual reward
        # for k in range(len(rewards)):

    if i_episode%save_interal==0:
            print(score/2000)
            score = 0
            torch.save(model.state_dict(), f'model_{i_episode}')


    if i_episode < 5:
        print("episode is %d " % i_episode, "num_experience is %d\n" % buff.num_experiences)
        continue

    for e in range(n_epoch):
        batch = buff.getBatch(batch_size)
        for j in range(batch_size):
            sample = batch[j]
            O[j] = sample[0]
            Next_O[j] = sample[3]
            Matrix[j] = sample[4]
            Next_Matrix[j] = sample[5]

        q_values = model(torch.Tensor(O), torch.Tensor(Matrix))
        target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim = 2)[0]
        target_q_values = np.array(target_q_values.data)
        expected_q = np.array(q_values.data)
        
        for j in range(batch_size):
            sample = batch[j]
            for i in range(n_ant-1):
                # print('debug',np.average(sample[2][i][0]) + (1-sample[6])*GAMMA*target_q_values[j][i])
                # print(j) 
                # print('sample',sample[2])
                # print('left',expected_q[j][i][sample[1][i]])
                expected_q[j][i][sample[1][i]] = sample[2] + (1-sample[6])*GAMMA*target_q_values[j][i] ## dimension problem 
        
        loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
        print('loss',loss)
        losses.append(loss.detach().numpy())
        # print(losses)
        np.save('loss.npy',losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    if i_episode%5 == 0:
        model_tar.load_state_dict(model.state_dict())
    


env.terminate()

