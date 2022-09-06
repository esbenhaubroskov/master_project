# Playground for RMP simulation 
# Inspired by Anqi Li
# @author Asbjørn Lybker and Esben Skov
# @date Februar 25, 2022

from rmp_tree_factory import DroneSwarmFactory
from environment_generator import EnvironmentGenerator
import events
from utils.environment_loader import EnvironmentTemplate

import time
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


from config import load_config
from plot import plot_acceleration, show_animation, Animate, plot_rmp_tree, OldAnimate
import system_solve
import metric
from itertools import combinations

# experiments\reward_shaping\config1\config_reward.json
# experiments\reward_shaping\config1\config_length.json

#config = load_config(r'experiments\residual_goal_attractor\config.json')
# Radio maximum distance:
#config = load_config(r'experiments\comm\comm_stretch_config.json')
#config = load_config(r'experiments\comm\comm_obstruction_config.json')

#config = load_config(r'experiments\comm_and_obstacles\corridor_config.json')

# Obstacle avoidance:
#config = load_config(r'experiments\hand_crafted_obstacle_avoidance\config.json')

# Residual goal attractor:
#config = load_config(r'experiments\residual_goal_attractor\config.json')

# Our residual obstacle:
#config = load_config(r'experiments\reward_shaping\config5\config_length.json')



#config = load_config(r'config.json')
fig_save_path = config.data_set['save_path']

config_string = json.dumps(config, indent=4, default=lambda x: "N/A")
metadata = dict(
    Description = config_string
)
gif_metadata = dict(
    comment = config_string
)
figure_format = ".pdf"
# --------------------------------------------
# ------------- Build RMP tree ---------------
# --------------------------------------------
print("Setting up environment")

if config.env_params["use_template"]:
    #env_file = open(Path(__file__).parent.resolve() / 'scenarios' / config.env_params["custom_env"])
    #custom_env = json.load(env_file)
    #env_file.close()
    #kwargs = custom_env["kwargs"]
    #formation_links = np.array(custom_env["formation_links"])
    #goal_links = custom_env["goal_links"]
    path = Path(__file__).parent.resolve() / 'scenarios' / config.env_params["custom_env"]
    template = EnvironmentTemplate(path, config)
    goal_links = template.goal_links
    formation_links = template.formation_links
    gen = EnvironmentGenerator(**template.kwargs)

else:
    gen = EnvironmentGenerator(config.uav_set["num_uavs"], 
                                config.env_params["n_obstacles"], 
                                config.env_params["obstacle_type"], 
                                config.env_params["drone_init_radius"], 
                                seed=config.env_params["seed"]
                                )
    formation_links = None
    goal_links = None

env_reset = not config.env_params["use_template"]
environment = gen.generate_environment(reset=env_reset)


swarm_factory = DroneSwarmFactory(config, environment, formation_links, goal_links)
print("Environment setup done")

# --------------------------------------------
# --------------- Simulating -----------------
# --------------------------------------------
x = np.array(swarm_factory.environment['starting_positions']).reshape(-1)
x_dot = np.zeros_like(x)
state_0 = np.concatenate((x, x_dot), axis=None)
r, leaf_dict = swarm_factory.get_drone_swarm()
robots = swarm_factory.robots

update_time = config.sim['update_time']
event_list = [events.TerminateAtArrival(swarm_factory.environment["goals"], config.sim["d_terminate"], config.sim["v_terminate"], goal_links=goal_links)]
swarm = system_solve.Drone_swarm(r, config, leaf_dict, state_0, events=event_list)
solver = system_solve.System_solver(swarm, state_0, [0, 200], update_time=update_time,
                                    atol=config.sim['atol'],rtol=config.sim['rtol'], method=config.sim['integrator'])

print(f"Seed: {gen.seed}")
print("Solving")
tic = time.time()
sol = solver.solve()
toc = time.time()
print(f"Solver done. Elapsed time: {toc-tic:.3f} s")
accs = swarm.accelerations


# ---------------------------------------------
# ---------- Animation and results ------------
# ---------------------------------------------
plt.rcParams.update({'font.size': 18}) # Before it was 16

child_accs = r.child_accs # Robot accelerations
root_childs = r.children # Robot leafs can be accessed

# Print metrics
arrivals = metric.get_arrival(sol, environment["goals"], goal_links=goal_links, vel_thres=config.sim["v_terminate"])
re = metric.print_metric(sol, arrivals)
metadata["Description"] += str(re)

# Plot RMP tree
if config.data_set['plot_rmp_tree']:
    plot_rmp_tree(r, graph=None, save_fig=config.data_set['save_rmp_tree'], save_path=config.data_set['save_path'])
    
# Plot accelerations for every robot leafs
labels = dict(
            ga = "Goal Attractor",
            md = "Max Distance", 
            ca = "Colision\nAvoidance",
            fc = "Formation Control",
            oa = "Circle-Based\nObstacle Avoidance",
            da = "Damper",
            po = "Polygon Obstacle",
            nn = "NN Obstacle\n Avoidance",
            lo = "Lidar Obstacle\nAvoidance",
            no = "Residual Obstacle\nAvoidance",
            gr = "Goal Attraction\nResidual",
            ro = "Drone"
        )

# For naming figures
abbreviations = {
    "max_distance": "md", 
    "radio_max_distance": "rmd",
    "lidar_obstacle_avoidance": "loa",
    "nn_obstacle_avoidance": "nnoa",
    "obstacle_avoidance": "oa",
    "collision_avoidance": "ca",
    "formation_control": "fc",
    "damper": "d",
    "goal_attractor": "ga", 
    "nn_obstacle_avoidance_residual": "nnoar",
    "nn_goal_residual": "nngr"
}

# Create policy summary strin
policy_summary = f"{config.uav_set['num_uavs']}"
for key, value in abbreviations.items():
    if config.uav_set[key]:
        policy_summary  += "_" + value
if config.env_params["use_template"]:
    policy_summary += "_" + config.env_params["custom_env"].rsplit( ".", 1 )[ 0 ]
else:
    policy_summary += f"_{config.env_params['n_obstacles']}_obst"

print(f'policy summary: {policy_summary}')

# Plot accelerations
i = 0
for a in accs[1:]: # start from 1 to exclude robot accelerations
    acc_save_path = config.data_set['save_path'] / f"{policy_summary}_acc_rob_{i}{figure_format}" 
    plot_acceleration(a, save_fig=config.data_set['save_accelerations_plot'], save_path=acc_save_path, labels=labels, metadata=metadata)
    i += 1

plot_frames = config.data_set["snapshot_frames"]

# Plot max allowed distance and RSSI
if config.uav_set["radio_max_distance"]:
    # Max allowed distance
    md_fig, md_ax = plt.subplots(num="Max distance")
    
    md_ax.set_xlabel("Time [s]")
    md_ax.set_ylabel("Maximum distance allowed [m]")
    for md in leaf_dict["radio_max_distance_controllers"]:
        drone_id = md.parent.name.split("_")[-1]
        md_ax.plot(sol.t, md.d_max_log, label=f"Drone {drone_id}")
    for frame in plot_frames:
        md_ax.axvline(frame*update_time, color='k', alpha=0.5)
        md_ax.annotate(f"t = {update_time*frame}", (frame*update_time+2, md_ax.get_ylim()[1]-1), rotation=-90)
    
    # Plot actual distances
    N_dim = 2
    n_states = sol.y.shape[-1]
    positions = [[] for _ in range(config.uav_set["num_uavs"])]
    for i in range(config.uav_set["num_uavs"]):
        positions[i] 
    for i in range(n_states):
        state = sol.y[:,i]
        state = state.reshape(2, -1)
        x = state[0]
        for j in range(config.uav_set["num_uavs"]):
            drone_x = x[N_dim*j:N_dim*j+2]
            positions[j].append(drone_x)
    combi_list = list(combinations([i for i in range(config.uav_set["num_uavs"])], 2))
    distance_vectors = [np.array(positions[x[0]]) - np.array(positions[x[1]]) for x in combi_list]
    distances_list = [[np.linalg.norm(d_vector) for d_vector in vectors] for vectors in distance_vectors]
    for i in range(len(distances_list)):
        md_ax.plot(sol.t, distances_list[i], label=r"Actual distance: $" + f"d_{{{combi_list[i][0]} {combi_list[i][1]}}}" + r"$")

    md_ax.legend()
    if config.data_set["save_max_distance"]:
        md_fig.savefig(fig_save_path / (policy_summary + "_max_dist" + figure_format), bbox_inches='tight', metadata=metadata)
    md_ax.set_title("Max distance based on radio signal quality")

    # RSSI
    rssi_fig, rssi_ax = plt.subplots(num="RSSI")
    rssi_ax.set_xlabel("Time [s]")
    rssi_ax.set_ylabel("RSSI [dB]")
    moving_averages = []
    lines = []
    for md in leaf_dict["radio_max_distance_controllers"]:
        drone_id = md.parent.name.split("_")[-1]
        lines.append(rssi_ax.plot(sol.t, md.rssi_log, alpha=0.5, label=f"Drone {drone_id}")[0])
        moving_averages.append(np.convolve(md.rssi_log, np.ones(30), 'same')/30)
    for i in range(len(lines)): # plot averages *after* the true RSSI values
        rssi_ax.plot(sol.t, moving_averages[i], color=lines[i].get_color())
    rssi_ax.legend()
    rssi_ax.axhline(config.radio_max_distance["rssi_critical"], color='k', alpha=0.5)
    #rssi_ax.axhline(config.radio_max_distance["rssi_no_signal"], color='r')
    rssi_ax.annotate("Desired minimum level", (5, config.radio_max_distance["rssi_critical"]+1.5))
    #rssi_ax.annotate("Loss of communication", (10, config.radio_max_distance["rssi_no_signal"]))
    for frame in plot_frames:
        rssi_ax.axvline(frame*update_time, color='k', alpha=0.5)
        rssi_ax.annotate(f"t = {update_time*frame}", (frame*update_time+2, rssi_ax.get_ylim()[1]-20), rotation=-90)
    if config.data_set["save_rssi"]:
        rssi_fig.savefig(fig_save_path / (policy_summary + "_rssi" + figure_format), bbox_inches='tight', metadata=metadata)
    rssi_ax.set_title("RSSI")

# Plot actual distances and max distance
if config.uav_set["max_distance"]:
    md_fig, md_ax = plt.subplots(num="Max distance")
    # Plot actual distances
    N_dim = 2
    n_states = sol.y.shape[-1]
    positions = [[] for _ in range(config.uav_set["num_uavs"])]
    for i in range(config.uav_set["num_uavs"]):
        positions[i] 
    for i in range(n_states):
        state = sol.y[:,i]
        state = state.reshape(2, -1)
        x = state[0]
        for j in range(config.uav_set["num_uavs"]):
            drone_x = x[N_dim*j:N_dim*j+2]
            positions[j].append(drone_x)
    combi_list = list(combinations([i for i in range(config.uav_set["num_uavs"])], 2))
    distance_vectors = [np.array(positions[x[0]]) - np.array(positions[x[1]]) for x in combi_list]
    distances_list = [[np.linalg.norm(d_vector) for d_vector in vectors] for vectors in distance_vectors]
    for i in range(len(distances_list)):
        md_ax.plot(sol.t, distances_list[i], label=r"Actual distance: $" + f"d_{{{combi_list[i][0]} {combi_list[i][1]}}}" + r"$")
    md_ax.axhline(config.max_distance["d_max"], color='k', alpha=0.5)
    md_ax.annotate("Desired maximum distance", (5, config.max_distance["d_max"]+0.2))
    md_ax.set_xlabel("Time [s]")
    md_ax.set_ylabel("Distance [m]")
    md_ax.legend()
    if config.data_set["save_max_distance"]:
        md_fig.savefig(fig_save_path / (policy_summary + "_max_dist" + figure_format), bbox_inches='tight', metadata=metadata)
    md_ax.set_title("Max distance")

# Plot speed over time
if config.data_set["plot_speed"]:
    speed_fig, speed_ax = plt.subplots(num="speed")
    #solved_xdots = [sol.y[i].reshape(2,-1)[2] for i in range(len(sol.y))]
    #solved_speeds = []

    N_dim = 2
    n_states = sol.y.shape[-1]
    velocities = [[] for _ in range(config.uav_set["num_uavs"])]
    for i in range(n_states):
        state = sol.y[:,i]
        state = state.reshape(2, -1)
        #x = state[0]
        x_dot = state[1]
        for j in range(config.uav_set["num_uavs"]):
            drone_x_dot = x_dot[N_dim*j:N_dim*j+2]
            velocities[j].append(np.linalg.norm(drone_x_dot))
    for i in range(config.uav_set["num_uavs"]):
        speed_ax.plot(sol.t, velocities[i], label=f"Drone {i}")
    speed_ax.set_xlabel("Time [s]")
    speed_ax.set_ylabel("Speed [m/s]")
    #speed_ax.legend()
    if config.data_set["save_speed"]:
        speed_fig.savefig(fig_save_path / (policy_summary + "_speed" + figure_format), bbox_inches='tight', metadata=metadata)

N = swarm_factory.N
x_g = swarm_factory.x_g
x_0 = swarm_factory.x_0
x_o = swarm_factory.x_o
r_o = swarm_factory.r_o    
poly = swarm_factory.poly
poly = environment["obstacle_polygons"]

# Reduce the font size for simulation
plt.rcParams.update({'font.size': 14}) # Before it was 10
# Creating animate object for animation
A = Animate(sol, N, x_g, x_0, x_o, r_o, poly, accs, update_time, swarm.t_log, labels, normalized_velocity=False, plot_arrows=False)
A.logarthmic_arrows = True

# Plot the scene for selected time steps
frame_figs = []
for frame in plot_frames:
    if config.data_set["include_max_distance"]:
        # Extract and plot ranges for max distance controllers
        md_ranges = [[] for _ in range(len(robots))]
        # Ranges for radio based max distance controllers
        for i in range(len(leaf_dict["radio_max_distance_controllers"])):
            md = leaf_dict["radio_max_distance_controllers"][i]
            drone_id = int(md.parent.name.split("_")[-1])
            md_ranges[drone_id].append(md.d_max_log[frame])
        # Ranges for static max distance controllers
        for i in range(len(leaf_dict["max_distance_controllers"])):
            md = leaf_dict["max_distance_controllers"][i]
            drone_id = int(md.parent.name.split("_")[-1])
            md_ranges[drone_id].append( config.max_distance["d_max"])
        #mds = [md.d_max_log[frame] for md in leaf_dict["radio_max_distance_controllers"]]
    
    # Plot the frame
    frame_fig, frame_ax = A.plot_frame(frame, md_ranges=md_ranges)
    frame_figs.append(frame_fig)
    file_name = policy_summary + f"_frame_{frame}" + figure_format
    
    
    # Save the frame
    if config.data_set["save_animation_frames"]:
        frame_fig.savefig(fig_save_path / file_name, bbox_inches='tight', metadata=metadata)
    frame_fig.show()
    

print("Animating")
# --------------------------------------------
show_animation(A, sol, config.data_set['save_path'], save_animation=config.data_set['save_animation'],
    metadata=gif_metadata, name=policy_summary)
print("Animation done")
plt.show()