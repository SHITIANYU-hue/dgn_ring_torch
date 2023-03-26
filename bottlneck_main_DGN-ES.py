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
from DGN_Env import para_produce_rl, Experiment
import logging

import datetime

import time

from DGN import DGN
from buffer import ReplayBuffer
from flow.core.params import SumoParams
### define some parameters
import pandas as pd
import torch.optim as optim


import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

from ES_VSL import ES_VSL, SGD
import multiprocessing as mp


#step1:train, step2:eval(result)
MODE = 'eval'

from config import *
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print(path+' '+'exist')
    return path

if MODE == 'train':
    num_runs = 100
elif MODE == 'eval':
    num_runs = 10
#changeable, must equal!!!  22  17  13  9  5
agent_num = 5
neighbors = 5
n_ant = agent_num

## define some environment parameters
exp_tag = "dgn_ring_ES"
result_dir = mkdir(f'{exp_tag}_kid_{N_KID}_results')
model_path = "./model/{0}_model.ckpt".format(exp_tag)
train_test=1 ##define train(1) or test(2)
## build up settings
flow_params = para_produce_rl(NUM_AUTOMATED=agent_num) # NUM_AUTOMATED=agent_num
env = Experiment(flow_params=flow_params).env

env.sim_params.emission_path='./{}_emission/'.format(exp_tag)
sim_params = SumoParams(sim_step=0.1, render=False, emission_path='./{0}_emission/'.format(exp_tag))
num_steps = env.env_params.horizon

 

buff = ReplayBuffer(capacity)
model = DGN(n_ant,observation_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,observation_space,hidden_dim,n_actions)
if MODE == 'train':
    model = model
    model_tar = model_tar
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
elif MODE == 'eval':
    model.load_state_dict(torch.load(result_dir + '/model_100'))

O = np.ones((batch_size,n_ant,observation_space))
Next_O = np.ones((batch_size,n_ant,observation_space))
Matrix = np.ones((batch_size,n_ant,n_ant))
Next_Matrix = np.ones((batch_size,n_ant,n_ant))

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


def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


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

def params_reshape(shapes, params):     # reshape to be a matrix
        p, start = [], 0
        for i, shape in enumerate(shapes):  # flat params to matrix
            n_w, n_b = shape[0] * shape[1], shape[1]
            p = p + [params[start: start + n_w].reshape(shape),
                    params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
            start += n_w + n_b
        return p



# utility instead reward for update parameters (rank transformation)
base = N_KID * 2    # *2 for mirrored sampling
rank = np.arange(1, base + 1)
util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
utility = util_ / util_.sum() - 1 / base
N_CORE = mp.cpu_count() - 1


SPEED_LIMITS = np.array([5, 10, 12, 15, 17, 20])
ESvsl = ES_VSL(observation_space, len(SPEED_LIMITS), N_KID, LR, SIGMA)
net_shapes, net_params = ESvsl.build_net()
VSL_optimizer = SGD(net_params, learning_rate=0.05)
pool = mp.Pool(processes=N_CORE)
mar = None

for i_episode in range(num_runs):
    print('episode is:',i_episode)
    

    ES_rewards=[]   # save the reward of VSL network
    total_distance = 0 # total distance that rl cars move
    

    # Evolution Strategy
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling
    for k_id in range(N_KID*2):
        if ((i_episode+1) % REFRESH_PERIOD != 0 or MODE == 'eval') and k_id != 0:  # refresh the speed limit every 10 episode
            continue                                    # but we still need to run DQN in the last loop (N_KID*2-1)
        print("k_id is: ", k_id)

        obs = env.reset()
        print("obs: ", obs)

        aset = [0] * agent_num
        aset_arg = [0] * agent_num
        score=0 
        # for temporary metrics
        average_speeds = []
        punish_accel = []
        headway_limit = []
        sp_limit = []
        
        if MODE == 'train':
            params = net_params
            seed = noise_seed[k_id]
            np.random.seed(seed)
            params += sign(k_id) * SIGMA * np.random.randn(params.size)
        elif MODE == 'eval':
            params = np.load(os.path.join(result_dir,'VSL_Params_100.npy'))
        
        p = params_reshape(net_shapes, params)  # convert the flatten to matrix
        
        
        veh_state = np.array(list(obs.values())).reshape(agent_num,-1)
        speed_limit = SPEED_LIMITS[ESvsl.get_action(p, veh_state)]
         
        print("speed_limit get action : ", speed_limit)

        arrive = 0
        max_outflow = 0
        comp_cnt = 0
        outflows = []
        speed_limits = []
        lane_vsl = []
        rl2id = {}
        for ind, rl_veh in enumerate(env.k.vehicle.get_rl_ids()):
            rl2id[rl_veh] = ind
        # Initialization for time_space_diagram
        time_pos_vel = {}
        cur_distance = {}
        lane_pos = {}
        for vel_id in env.k.vehicle.get_ids():
            time_pos_vel[vel_id] = np.zeros((2, num_steps))     # 0: position 1: velocity
            cur_distance[vel_id] = env.k.vehicle.get_x_by_id(vel_id)
            cur_lane = env.k.vehicle.get_lane(vel_id)
            lane_pos[cur_lane] = {x : np.zeros((num_steps)) for x in env.k.vehicle.get_ids()}

        for j in range(num_steps):
            # manager actions
            # convert state into values
            state_ = np.array(list(obs.values())).reshape(agent_num,-1).tolist()

            adj, headways = Adjacency(env ,neighbors=neighbors)

            state_= torch.tensor(np.asarray([state_]),dtype=torch.float) 

            adj_= torch.tensor(np.asarray(adj),dtype=torch.float)


            
            q = model(state_, adj_)[0]
            for i in range(n_ant):
                if np.random.rand() > epsilon:
                    a = 3*np.random.randn(n_actions)
                else:
                    a = q[i].argmax().item()
                    aset_arg[i] = a
                    action_lists = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15] 
                    a = action_lists[a]
                    aset[i] = a

            action_dict = {}
            k=0
            for key, value in obs.items():       
                action_dict[key]=aset[k]         
                k+=1

            if MODE == 'train':
                if i_episode % REFRESH_PERIOD == 0:             # refresh the speed_limit every 10 episode
                    speed_limit_ = speed_limit
            elif MODE == 'eval':
                if j % REFRESH_PERIOD == 0:                     # refresh the speed_limit every 10 horizon
                    veh_state = np.array(list(obs.values())).reshape(agent_num,-1)
                    speed_limit_ = SPEED_LIMITS[ESvsl.get_action(p, veh_state)]
                    speed_limits.append(speed_limit)

                # speed_limit_ = [10] * agent_num
            


            
            next_state, reward, done, _, metrics = env.step(action_dict, speed_limit_)
            next_adj, next_headways = Adjacency(env ,neighbors=neighbors)


            while len(next_state) < agent_num:              # padding the matrix to maintain dimension
                next_state['comp_veh_{}'.format(comp_cnt)] = np.array([0,0,0])
                comp_cnt += 1


            next_state_ = np.array(list(next_state.values())).reshape(agent_num,-1).tolist()

            if done["__all__"]: 
                done_ = 1
            else:
                done_ = 0
            
            if done_ == 1:
                reward_ = [-5]*agent_num
            else:
                reward_ = np.array(list(reward.values())).reshape(1,-1).tolist()

            
            buff.add(np.array(state_),aset_arg,np.average(reward_),np.array(next_state_),np.array(adj[-1]),np.array(next_adj[-1]), done_)
            obs = next_state

            # calculate the car flow, all the cars
            outflow = env.k.vehicle.get_outflow_rate(500)
            max_outflow = max(max_outflow, outflow)
            arrive += len(env.k.vehicle.get_arrived_ids())
            target_headway, time_headway = calculate_hw(env)
            #average_speeds.append(metrics[0])
            punish_accel.append(metrics[1])
            headway_limit.append(metrics[2])
            sp_limit.append(metrics[3])
            outflows.append(outflow)
            ACCEL.append(aset)
            TARGET_HW.append(target_headway)
            TIME_HW.append(time_headway)

            cur_speed = calculate_aver_speed(env)
            if cur_speed > 0 :
                average_speeds.append(cur_speed)
            if MODE == 'eval':
                for i, vel_id in enumerate(time_pos_vel.keys()):
                    if vel_id in env.k.vehicle.get_ids():       # cars still in road
                        time_pos_vel[vel_id][1][j] = env.k.vehicle.get_speed(vel_id)
                        cur_distance[vel_id] += env.k.vehicle.get_speed(vel_id) * j * 0.0001   # velocity * time
                        # save position in lane
                        cur_lane = env.k.vehicle.get_lane(vel_id)
                        if cur_lane in lane_pos.keys() and vel_id in lane_pos[cur_lane].keys():
                            lane_pos[cur_lane][vel_id][j] = cur_distance[vel_id]

                    else:                                       # cars reach the end
                        time_pos_vel[vel_id][1][j] = 0
                    
                    time_pos_vel[vel_id][0][j] = cur_distance[vel_id]
                
                # calcualte average speed limit of each lane
                aver_vsl = [0] * 4
                lane_num_rlveh = [0] * 4
                for rl_veh in env.k.vehicle.get_rl_ids():
                    cur_lane = env.k.vehicle.get_lane(rl_veh)
                    if cur_lane > 0:        # prevent negative value of lane caused by car crash
                        aver_vsl[cur_lane] += speed_limit_[rl2id[rl_veh]]
                        lane_num_rlveh[cur_lane] += 1

                for road_lane in range(4):
                    if lane_num_rlveh[road_lane] > 0:
                        aver_vsl[road_lane] /= lane_num_rlveh[road_lane]
                lane_vsl.append(aver_vsl)



            if done_ == 1:
                score += -5*agent_num
            else:
                score += sum(list(reward.values()))
            
            if done_ == 1:
                print("================================================================================")
                print("Crash!!!!")
                print("================================================================================")
                # total_distance = sum([env.k.vehicle.get_x_by_id(rl_id) for rl_id in env.k.vehicle.get_rl_ids()])
                # total_distance += arrive * LANE_DISTANCE
                car_crash += 1
                break
            
            if len(env.k.vehicle.get_rl_ids()) == 0:        # all the cars reach the destination
                # total_distance = agent_num * LANE_DISTANCE
                break; 


            if j % 100 == 0:
                print("j : ", j)
                print("outflow : ", outflow)
                print("len of arrive id : ", arrive)
                print("max_outflow", max_outflow)
                print("action dict", action_dict)
            

        # aver_speed = calculate_aver_speed(env)
        ES_rewards.append(sum(outflows) / len(outflows))
        OUTFLOWS.append(outflows)   
        if len(average_speeds) > 0 : AVERAGE_SPEED.append(np.mean(average_speeds))
        PUNISH_ACCEL.append(np.mean(punish_accel))
        HEADWAY_LIMIT.append(np.mean(headway_limit))
        SP_LIMIT.append(np.mean(sp_limit))
        ARRIVE.append(arrive)
        ACCEL.append(aset)
        ES_TOTAL_SPL.append(speed_limits)
        total_distances.append(total_distance)
        LANE_VSL.append(lane_vsl)

        time_pos_vel_nd = np.zeros((22, 2, num_steps))
        for i, vel_id in enumerate(time_pos_vel.keys()):
            time_pos_vel_nd[i] = time_pos_vel[vel_id]
        TIME_POS_VEL.append(time_pos_vel_nd)
        LANE_POS.append(lane_pos)

        scores.append(score/num_steps)
        car_crashs.append(car_crash)
        
    
    
    ES_rewards = np.array(ES_rewards)


    ES_TOTAL_SCORES.append(ES_rewards.mean())

    
    

    np.save(os.path.join(result_dir, 'scores.npy'),scores)
    np.save(os.path.join(result_dir, 'ES_speed_limit.npy'), ES_TOTAL_SPL)
    np.save(os.path.join(result_dir, 'ES_Total_scores.npy'), ES_TOTAL_SCORES)
    np.save(os.path.join(result_dir, 'car_crashs.npy'), car_crashs)
    np.save(os.path.join(result_dir,'arrive_cars.npy'), ARRIVE)
    np.save(os.path.join(result_dir,'rl_car_accel.npy'), ACCEL)
    #np.save('car_total_distance.npy', total_distances)
    np.save(os.path.join(result_dir, 'average_speed.npy'), AVERAGE_SPEED)
    np.save(os.path.join(result_dir, 'punish_accel.npy'), PUNISH_ACCEL)
    np.save(os.path.join(result_dir, 'headway_limit.npy'), HEADWAY_LIMIT)
    np.save(os.path.join(result_dir, 'sp_limit.npy'), SP_LIMIT)
    np.save(os.path.join(result_dir, 'time_pos_vel.npy'), TIME_POS_VEL)
    np.save(os.path.join(result_dir, 'lane_pos.npy'), LANE_POS)
    np.save(os.path.join(result_dir, 'outflows.npy'), OUTFLOWS)
    np.save(os.path.join(result_dir, 'lane_vsl.npy'), LANE_VSL)

        
    if MODE == 'train':

        if (i_episode+1) % REFRESH_PERIOD == 0:                    # train the VSL network every 10 episode

            kids_rank = np.argsort(ES_rewards)[::-1]               # rank kid id by reward

            
            cumulative_update = np.zeros_like(net_params)       # initialize update values
            for ui, k_id in enumerate(kids_rank):
                np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
                cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)

            gradients = VSL_optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))

            net_params += gradients
            kid_rewards = ES_rewards
            # save the parameters of VSL Network
            np.save(os.path.join(result_dir, f'VSL_Params_{i_episode+1}.npy'), net_params)
            print(
                'Gen: ', i_episode,
                #'| Net_R: %.1f' % mar,
                '| Kid_avg_R: %.1f' % kid_rewards.mean(),
            )

        
        if (i_episode+1) % save_interal==0:
                print(score/2000)
                score = 0
                torch.save(model.state_dict(), os.path.join(result_dir, f'model_{i_episode+1}'))
                


        if i_episode < 5:
            # print("episode is %d " % i_episode, "num_experience is %d\n" % buff.num_experiences)
            continue

        


        for e in range(n_epoch):
            batch = buff.getBatch(batch_size)
            for j in range(len(batch)):
                # (obs, action, reward, new_obs, matrix, next_matrix, done)
                sample = batch[j]
                O[j] = sample[0]
                Next_O[j] = sample[3]
                Matrix[j] = sample[4]
                Next_Matrix[j] = sample[5]

            q_values = model(torch.Tensor(O), torch.Tensor(Matrix))
            target_q_values = model_tar(torch.Tensor(Next_O), torch.Tensor(Next_Matrix)).max(dim = 2)[0]
            target_q_values = np.array(target_q_values.data)
            expected_q = np.array(q_values.data)
            
            for j in range(len(batch)):
                sample = batch[j]
                for i in range(n_ant-1):
                    expected_q[j][i][sample[1][i]] = sample[2] + (1- sample[6])*GAMMA*target_q_values[j][i] ## dimension problem 
            
            loss = (q_values - torch.Tensor(expected_q)).pow(2).mean()
            print("loss : ", loss)
            losses.append(loss.detach().numpy())
            # print(losses)
            np.save(os.path.join(result_dir, 'loss.npy'),losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if i_episode%5 == 0:
            model_tar.load_state_dict(model.state_dict())
    


env.terminate()

