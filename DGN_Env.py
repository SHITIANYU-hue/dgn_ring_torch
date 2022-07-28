# Import all of the necessary pieces of Flow to run the experiments
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    InFlows, SumoLaneChangeParams, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.core.params import TrafficLightParams

 
from flow.controllers import RLController, IDMController, ContinuousRouter
from flow.envs.ring.wave_attenuation import WaveAttenuationEnv, WaveAttenuationPOEnv # Env for RL
from flow.envs.multiagent import MultiAgentWaveAttenuationPOEnv, MultiAgentWaveAttenuationPOEnvBN
from flow.core.experiment import Experiment
from flow.envs import BottleneckEnv
from flow.networks import BottleneckNetwork
import logging

import datetime
import numpy as np
import time
import os

from flow.core.util import emission_to_csv
from flow.utils.registry import make_create_env
    
# define parameters


from flow.controllers.rlcontroller import RLController
from flow.controllers.lane_change_controllers import SimLaneChangeController
from flow.controllers.routing_controllers import ContinuousRouter
from copy import deepcopy
from gym.spaces.box import Box
from flow.core import rewards
from flow.envs.base import Env

#flow/examples/exp_configs/rl/singleagent/

def para_produce_rl(HORIZON = 3000, NUM_AUTOMATED = 4):
	
    # time horizon of a single rollout
    HORIZON = 3000
    # number of rollouts per training iteration
    N_ROLLOUTS = 20
    # number of parallel workers
    N_CPUS = 2
    # number of automated vehicles. Must be less than or equal to 22.
    NUM_AUTOMATED = NUM_AUTOMATED

    # bottleneck
    INFLOW = 2300

    DISABLE_TB = True
    # If set to False, ALINEA will control the ramp meter
    DISABLE_RAMP_METER = True

    SCALING = 1

    AV_FRAC = 0.10
    # We evenly distribute the automated vehicles in the network.
    num_human = 22 - NUM_AUTOMATED
    humans_remaining = num_human

    vehicles = VehicleParams()
    for i in range(NUM_AUTOMATED):
        # Add one automated vehicle.
        vehicles.add(
            veh_id="rl_{}".format(i),
            # veh_id="rl",
            acceleration_controller=(RLController, {}),
            lane_change_controller=(SimLaneChangeController, {}),
            routing_controller=(ContinuousRouter, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="aggressive",
            ),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=1)

        # Add a fraction of the remaining human vehicles.
        vehicles_to_add = round(humans_remaining / (NUM_AUTOMATED - i))
        humans_remaining -= vehicles_to_add
        vehicles.add(
            veh_id="human_{}".format(i),
            #veh_id="human",
            acceleration_controller=(IDMController, {
                "noise": 0.2
            }),
            lane_change_controller=(SimLaneChangeController, {}),
            car_following_params=SumoCarFollowingParams(
                speed_mode="all_checks",
            ),
            routing_controller=(ContinuousRouter, {}),
            lane_change_params=SumoLaneChangeParams(
                lane_change_mode=0,
            ),
            num_vehicles=vehicles_to_add)

        inflow = InFlows()
        inflow.add(
            veh_type="human_{}".format(i),
            edge="1",
            vehs_per_hour=INFLOW * (1 - AV_FRAC),
            depart_lane="random",
            depart_speed=10)
        
        inflow.add(
            veh_type="rl_{}".format(i),
            edge="1",
            vehs_per_hour=INFLOW * AV_FRAC,
            depart_lane="random",
            depart_speed=10)

        flow_params = dict(
        # name of the experiment
        exp_tag="multiagent_bottleneck",

        # name of the flow environment the experiment is running on
        env_name=MultiAgentWaveAttenuationPOEnvBN,  # MultiAgentWaveAttenuationPOEnv

        # name of the network class the experiment is running on
        network=BottleneckNetwork,

        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=True,
            restart_instance=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            warmup_steps=750,
            clip_actions=False,
            additional_params={
                "max_accel": 1,
                "max_decel": 1,
                "ring_length": [220, 270],
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            inflows=inflow,
            additional_params={
                "scaling": 1,
                "speed_limit": 23,
                "lanes": 1,
                "resolution": 40,
            }, ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(
            spacing="random",
            min_gap=5,
            lanes_distribution=float("inf"),
            edges_distribution=["2", "3", "4", "5"]
        ))

    flow_params['env'].horizon = HORIZON
    return flow_params


#flow_params = para_produce(flow_rate=1000, scaling=1, disable_tb=True, disable_ramp_meter=True)
flow_params = para_produce_rl()
	
class Experiment:

    def __init__(self, flow_params=flow_params):
        """Instantiate Experiment."""
        # Get the env name and a creator for the environment.
        create_env, _ = make_create_env(flow_params)

        # Create the environment.
        self.env = create_env()

        logging.info(" Starting experiment {} at {}".format(
            self.env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")	
