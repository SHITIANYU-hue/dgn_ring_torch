import os
import numpy as np
import torch

# for reproducing
args_seed = 64
torch.manual_seed(args_seed)
torch.cuda.manual_seed(args_seed)
torch.cuda.manual_seed_all(args_seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args_seed)
np.random.seed(args_seed)

# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams
import pandas as pd
from flow.controllers import SimLaneChangeController, ContinuousRouter
from flow.core.experiment import Experiment

import logging

import datetime

import time
import sys
sys.path.append('/home/jianshuaifeng/fengjs/dgn_ring_torch')

from DGN import DGN
from DGN_Env import para_produce_rl, Experiment
from buffer import ReplayBuffer
from flow.core.params import SumoParams
### define some parameters
import pandas as pd
import torch.optim as optim


import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import multiprocessing as mp


from config import *
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+'exist')
    
    return path


agent_num = 5
neighbors = 5
n_ant = agent_num
## define some environment parameters
exp_tag = "only_gipps"
results_dir = mkdir('{}_results'.format(exp_tag))

print(results_dir)
train_test=1 ##define train(1) or test(2)
## build up settings
flow_params = para_produce_rl(NUM_AUTOMATED=agent_num) # NUM_AUTOMATED=agent_num
env = Experiment(flow_params=flow_params).env

env.sim_params.emission_path='./{}_emission/'.format(exp_tag)
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='./{0}_emission/'.format(exp_tag))
num_steps = env.env_params.horizon



# buff = ReplayBuffer(capacity)
# model = DGN(n_ant,observation_space,hidden_dim,n_actions)
# model_tar = DGN(n_ant,observation_space,hidden_dim,n_actions)
# model = model
# model_tar = model_tar
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)

# O = np.ones((batch_size,n_ant,observation_space))
# Next_O = np.ones((batch_size,n_ant,observation_space))
# Matrix = np.ones((batch_size,n_ant,n_ant))
# Next_Matrix = np.ones((batch_size,n_ant,n_ant))

t = time.time()


# compension count

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
        x_pos = np.array([env.k.vehicle.get_x_by_id(veh_id) for veh_id in env.k.vehicle.get_rl_ids()])
        exist_agent_num = len(x_pos)
            
        while len(x_pos) < agent_num:               # rl vehs reach the end, we should maintain the dim of array
            x_pos = np.append(x_pos, 0)

        headways = np.zeros([len(x_pos), len(x_pos)])
        for d in range(len(x_pos)):
            headways[d,:] = abs(x_pos-x_pos[d])
        
        #print("headways : ", headways)
        
        orders = np.argsort(headways)

        #print("orders : ", orders)

        for _ in range(len(x_pos)):
            l = np.zeros([neighbors,len(x_pos)])
            for k in range(neighbors):   # original range(neighbours)
                # modify this condition to define the adjacency matrix
                l[k,orders[k]]=1

            adj.append(l)
        return adj, headways

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
    
    if aver_speed > 0 : aver_speed /= len(env.k.vehicle.get_ids())
    #print("aver_speed : ",aver_speed)
    return aver_speed

def calculate_hw(env):
    target_hws = []
    time_hws = []
    for veh_id in env.k.vehicle.get_ids():
        speed = env.k.vehicle.get_speed(veh_id)
        lead_veh = env.k.vehicle.get_leader(veh_id)
        # print("lead_veh :", lead_veh)
        if lead_veh is None:
            target_hw = np.nan
            time_hw = np.nan
        else:
            gap = env.k.vehicle.get_headway(veh_id)
            lead_veh_speed = env.k.vehicle.get_speed(lead_veh)
            if speed > lead_veh_speed:
                target_hw=0.1+0.5*(speed*speed-lead_veh_speed*lead_veh_speed)/(3*speed)
            else:
                target_hw=0.1

            # print("target_hw : ", target_hw)
            if speed > 0:
                time_hw = gap / speed
            else:
                time_hw = np.nan
            # print("time_hw : ", time_hw)
            
            target_hws.append(target_hw)
            time_hws.append(time_hw)
    
    if len(target_hws) == 0 :            # prevent there is no rl
        aver_target_hw = np.nan
        aver_time_hw = np.nan
    else:
        aver_target_hw = sum(target_hws) / len(target_hws)
        aver_time_hw = sum(time_hws) / len(time_hws)
    # print("aver_target_hw : ", aver_target_hw)
    # print("aver_time_hw : ", aver_time_hw)
    return aver_target_hw, aver_time_hw


def gipps_constrain_action(state):
    veh_speed, rel_speed, distance_headway = np.split(state, 3, axis=1)

    tau=0.1
    b=-1
    b_l=-1
    h=distance_headway
    s0=2
    acc=1.5
    sim_step=1
    v_l=veh_speed+rel_speed
    speed_limit =  10
    # get velocity dynamics
    v_acc = veh_speed + (2.5 * acc * tau * (
            1 - veh_speed) * np.sqrt(0.025 + veh_speed))
    v_safe = (tau * b) + np.sqrt(((tau**2) * (b**2)) - (
            b * ((2 * (h-s0)) - (tau * veh_speed) - ((v_l**2) / b_l))))
    #print('v safe',v_safe,'v_acc',v_acc)
    # v_next = min(v_acc, v_safe, speed_limit)
    if max(v_safe) > speed_limit:
        v_next = [speed_limit] * agent_num
    else:
        v_next = v_acc
    #print('veh speed',veh_speed)
    accels =(v_next-veh_speed)/sim_step
    #print('accel',accels)
    if accels[0][0]>3:
        accels=[[3]]
    if accels[0][0]<-3:
        accels=[[-3]]
    return accels


for i_episode in range(num_runs):
    # logging.info("Iter #" + str(i))
    print('episode is:',i_episode)
    aset = [0] * agent_num
    aset_arg = [0] * agent_num
    obs = env.reset()
    print("obs: ", obs)
    score=0 

    
    total_distance = 0 # total distance that rl cars move
    # for temporary metrics
    average_speeds = []
    punish_accel = []
    headway_limit = []
    speed_limit = [10] * agent_num

    comp_cnt = 0
    arrive = 0
    max_outflow = 0
    
    outflows = []
    for j in range(num_steps):
        # manager actions
        # convert state into values
        state_ = np.array(list(obs.values())).reshape(agent_num,-1)

        adj, headways = Adjacency(env ,neighbors=neighbors)

        # state_= torch.tensor(np.asarray([state_]),dtype=torch.float) 

        # adj_= torch.tensor(np.asarray(adj),dtype=torch.float)


            
        # q = model(state_, adj_)[0]
        actions = gipps_constrain_action(state_)
        # print("actions", actions)

        # for i in range(n_ant):
        #     if np.random.rand() > epsilon:
        #         a = 3*np.random.randn(n_actions)
        #     else:
        #         a = actions
        #         aset_arg[i] = a
        #         action_lists = [-0.1, -0.05, 0, 0.05, 0.1, 0.15] 
        #         a = action_lists[a]
        #         aset[i] = a

        action_dict = {}
        k=0
        for key, value in obs.items():       
            action_dict[key] = actions[k]         
            k+=1


        next_state, reward, done, _, metrics = env.step(action_dict, speed_limit)
        next_adj, next_headways = Adjacency(env ,neighbors=neighbors)

        while len(next_state) < agent_num:              # padding the matrix to maintain dimension
            next_state['comp_veh_{}'.format(comp_cnt)] = np.array([0,0,0])
            comp_cnt += 1


        next_state_ = np.array(list(next_state.values())).reshape(agent_num,-1).tolist()

        if done['__all__']:     # crash
            done_ = 1
        else:
            done_ = 0
        
        if done_ == 1:
            reward_ = [-5]*agent_num
        else:
            reward_ = np.array(list(reward.values())).reshape(1,-1).tolist()
            
        
        # buff.add(np.array(state_),aset_arg,np.average(reward_),np.array(next_state_),np.array(adj[-1]),np.array(next_adj[-1]), done_)
        obs = next_state


        # calculate the car flow, all the cars
        outflow = env.k.vehicle.get_outflow_rate(500)
        max_outflow = max(max_outflow, outflow)
        arrive += len(env.k.vehicle.get_arrived_ids())
        target_headway, time_headway = calculate_hw(env)
        # print("target_headway : ", target_headway)
        # print("time_headway : ", time_headway)
        #average_speeds.append(metrics[0])
        punish_accel.append(metrics[1])
        headway_limit.append(metrics[2])
        outflows.append(outflow)
        ACCEL.append(actions)
        
        TARGET_HW.append(target_headway)
        TIME_HW.append(time_headway)

        cur_speed = calculate_aver_speed(env)
        if cur_speed > 0 :
            average_speeds.append(cur_speed)


        if done_ == 1:
            score += -5*agent_num
        else:
            score += sum(list(reward.values()))
            
        if done_ == 1:                                  # crash
            print("================================================================================")
            print("Crash!!!!")
            print("================================================================================")
            car_crash += 1
            # total_distance = sum([env.k.vehicle.get_x_by_id(rl_id) for rl_id in env.k.vehicle.get_rl_ids()])
            # total_distance += arrive * LANE_DISTANCE
            break

        if len(env.k.vehicle.get_rl_ids()) == 0:        # all the cars reach the destination
            #total_distance = agent_num * LANE_DISTANCE
            break;
     

        if j % 100 == 0:
            print("j : ", j)
            print("outflow : ", outflow)
            print("len of arrive id : ", arrive)
            print("max_outflow", max_outflow)
            print("action dict", action_dict)
            

    
    # aver_speed = calculate_aver_speed(env)
    throughput.append(sum(outflows) / len(outflows))        # set threshold of 500 outflow  
    OUTFLOWS.append(outflows)
    ARRIVE.append(arrive)
    if len(average_speeds) > 0 : AVERAGE_SPEED.append(np.mean(average_speeds))
    PUNISH_ACCEL.append(np.mean(punish_accel))
    HEADWAY_LIMIT.append(np.mean(headway_limit))
    # total_distances.append(total_distance)


    scores.append(score/num_steps)
    car_crashs.append(car_crash)

    np.save(os.path.join(results_dir,'scores.npy'),scores)
    np.save(os.path.join(results_dir,'ES_Total_scores.npy'), throughput)
    np.save(os.path.join(results_dir,'car_crashs.npy'), car_crashs)
    np.save(os.path.join(results_dir,'arrive_cars.npy'), ARRIVE)
    np.save(os.path.join(results_dir,'rl_car_accel.npy'), ACCEL)
    np.save(os.path.join(results_dir,'target_hw.npy'), TARGET_HW)
    np.save(os.path.join(results_dir,'TIME_hw.npy'), TIME_HW)
    # np.save(os.path.join(results_dir,'car_total_distance.npy'), total_distances)
    np.save(os.path.join(results_dir,'average_speed.npy'), AVERAGE_SPEED)
    np.save(os.path.join(results_dir,'punish_accel.npy'), PUNISH_ACCEL)
    np.save(os.path.join(results_dir,'headway_limit.npy'), HEADWAY_LIMIT)
    np.save(os.path.join(results_dir,'outflows.npy'), OUTFLOWS)


    # if i_episode%save_interal==0:
    #     print(score/2000)
    #     score = 0
    #     torch.save(model.state_dict(), os.path.join(results_dir,'model_{}'.format(i_episode)))


    # if i_episode < 5:
    #     # print("episode is %d " % i_episode, "num_experience is %d\n" % buff.num_experiences)
    #     continue

    


    # for e in range(n_epoch):
    #     batch = buff.getBatch(batch_size)
    #     for j in range(batch_size):
    #         sample = batch[j]
    #         O[j] = sample[0]
    #         Next_O[j] = sample[3]
    #         Matrix[j] = sample[4]
    #         Next_Matrix[j] = sample[5]

    #     q_values = model(torch.Tensor(O), torch.Tensor(Matrix))
    #     target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim = 2)[0]
    #     target_q_values = np.array(target_q_values.data)
    #     expected_q = np.array(q_values.data)
        
    #     for j in range(batch_size):
    #         sample = batch[j]
    #         for i in range(n_ant-1):
    #             expected_q[j][i][sample[1][i]] = sample[2] + (1- sample[6])*GAMMA*target_q_values[j][i] ## dimension problem 
            
    #     loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
    #     print("loss : ", loss)
    #     losses.append(loss.detach().numpy())
    #     # print(losses)
    #     np.save(os.path.join(results_dir,'loss.npy'),losses)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()


    # if i_episode%5 == 0:
    #     model_tar.load_state_dict(model.state_dict())
    


env.terminate()

