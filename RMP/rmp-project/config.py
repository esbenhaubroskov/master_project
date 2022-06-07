import os, os.path
from pathlib import Path
import json
#from utils.formatter import dotdict
from utils.formatter import DotDict
verbose = True


f = open(Path(__file__).parent.resolve() / 'config.json')

config = json.load(f)
# Closing file
f.close()

config = DotDict(config)
config.learning['model_path'] = Path(__file__).parent.resolve() / config.learning['model_path']
config.data_set['save_path'] = Path(__file__).parent.resolve() / config.data_set['save_path']

def load_config(config_rel_path):
    f = open(Path(__file__).parent.resolve() / config_rel_path)

    config = json.load(f)
    # Closing file
    f.close()

    config = DotDict(config)
    config.learning['model_path'] = Path(__file__).parent.resolve() / config.learning['model_path']
    config.data_set['save_path'] = Path(__file__).parent.resolve() / config.data_set['save_path']
    return config

if __name__ == '__main__':
    f = open(Path(__file__).parent.resolve() / 'config.json')

    config = json.load(f)
    # Closing file
    f.close()

    config = DotDict(config)
    config.learning['model_path'] = Path(__file__).parent.resolve() / config.learning['model_path']
    config.data_set['save_path'] = Path(__file__).parent.resolve() / config.data_set['save_path']
    #print(config.learning['model_path'], config.data_set['save_path'])
 






'''
# --- Assign Policies to UAV --- 
uav_set = dict(
    #dynamic_leaf = False,
    num_uavs = 1, 
    max_distance = False,
    obstacle_avoidance = True,
    nn_obstacle_avoidance = True,
    collision_avoidance = False,
    goal_attractor = True,
    formation_control = False,
    damper = False,
    radio_max_distance = True,
)

# --- Setup sensors ---
lidar = dict( # TODO: use this when initalizing the Lidar class.
    n_rays = 20,
    max_range = 5,
)

# --- Assigning learning attributs --- 
learning = dict(
    train = False,
    model_path = Path(__file__).parent.resolve() / "results/model.pt",
    network_type = "ppo", # 'ppo'(Proximal Policy Gradient), 'pg'(policy gradient / REINFORCE)
    device = 'cpu',
    discount_factor = 0.99,
    num_episodes = 1000000, 
    max_steps = 500,
    batch_size = 1600,
    solved_score = 195,#score needed for environement to be considered solved
    goal_gain = 30,
    collision_gain = 30,
    distance_gain = 0.001,
    control_gain = 0.05,
    clearance_gain = 0.001,
    min_reward = -100,
    max_reward = 100,
)

# --- Enviroment setup ---
env_params = dict(
    n_obstacles = 2,
    obstacle_type = "circles",
    drone_init_radius = 2.5,
    seed = 6033072237823465043 #8750536495944257695
)

# --- Data plot and save settings ---
data_set = dict(
    save_path = Path(__file__).parent.resolve() / "results" ,
    save_animation = False,
    save_accelerations_plot = False,
    plot_accelerations = True,
    plot_rmp_tree = False,
    save_rmp_tree = False,
)


# --- Simulation settings ---
sim = dict(
    atol=1e-1,
    rtol=1e-1,
    d_terminate=0.2,
    v_terminate=1,
)

# --- Policy hyperparameters ---
max_distance = dict(
    d_max = 4,
    w = 1,
    alpha = 3,
    beta = 2,
)

goal_attractor = dict(
    alpha = 1,
    gain = 1,  
    eta = 2,
)

obstacle_avoidance = dict(
    epsilon=0.2,
)

poly_obst_avoid = dict(
    d_min = 0.5,
    w=1,
    alpha=1,
    eta=1,
    epsilon=1
)

collision_avoidance = dict( # Decentrialized
    R = 0.3,
    eta = 10,
)

nn_collision_avoidance = dict(
    deterministic=False,
)

formation_flying = dict(
    formation_type = "triangle", # square, line, grid - not setup in main file
    weight_fc = 5,
    dist = 1,
)

damper = dict(
    w = 2,
    eta = 1.2,
)

if __name__ == '__main__':
    import pandas as pd
    config_data = pd.DataFrame()
'''