'''
Author: SHITIANYU-hue tianyu.s@outlook.com
Date: 2022-06-26 06:43:58
LastEditors: SHITIANYU-hue tianyu.s@outlook.com
LastEditTime: 2022-08-31 12:50:00
FilePath: /dgn_ring_torch/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
hidden_dim = 128

max_step = 500
GAMMA = 0.99
n_episode = 1000000
i_episode = 0
capacity = 1000000
batch_size = 128
n_epoch = 5
epsilon = 0.9
score = 0


build_adj = 2

num_runs = 100
rl_actions = None
convert_to_csv = True



observation_space = 3
n_actions = 6

save_interal = 20
rets = []
mean_rets = []
ret_lists = []
vels = []
mean_vels = []
std_vels = []
times = []
vehicle_times = []
ploss = 0
qloss = 0
reg_loss = 0
results = []
scores = []
losses = []
car_crash = 0   # number of car crash
car_crashs = []
total_distances = []
LANE_DISTANCE = 735

# ES Parameters
# Evolution Strategy Vehicle Speed Limit

#changable!!!
N_KID = 2             # 2 4 8
LR = .05                     # learning rate
SIGMA = .05                 # mutation strength or step size

ES_TOTAL_SPL = []
ES_TOTAL_SCORES = []
REFRESH_PERIOD = 10
throughput=[]   # save the reward of VSL network (outflow)
ARRIVE = []     # save the number of arrive car in each episode
ACCEL = []      # save the accel of rl cars
TARGET_HW = []  # save the target headways of all the cars
TIME_HW = []    # save the time headways of all the cars
TIME_POS_VEL = [] # save the time-pos-vel data
LANE_POS = []   # save the lane the vehicles on
OUTFLOWS = []
# Reward Metrics
AVERAGE_SPEED = []
PUNISH_ACCEL = []
HEADWAY_LIMIT = []
SP_LIMIT = []
LANE_VSL = []
