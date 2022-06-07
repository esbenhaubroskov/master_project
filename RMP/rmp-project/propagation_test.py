from matplotlib.pyplot import xlabel, ylabel


if __name__ == "__main__":
    import radio.propagation
    import sys
    from pathlib import Path
    from environment_generator import EnvironmentGenerator
    from rmp_tree_factory import DroneSwarmFactory
    import system_solve
    from config import config
    import numpy as np
    import matplotlib.pyplot as plt
    import shapely as sly
    import shapely.ops
    from shapely.geometry import Point
    import events
    from plot import show_animation, Animate
    
    update_time = 0.2
    config.uav_set= {
        "num_uavs": 3, 
        "max_distance": False,
        "radio_max_distance": True,
        "obstacle_avoidance": True,
        "nn_obstacle_avoidance": False,
        "collision_avoidance": True,
        "goal_attractor": True,
        "formation_control": False,
        "damper": False
    }


    gen = EnvironmentGenerator(2, 6, "circles", 8, xlim=[-20,20], ylim=[-20,20])
    env = gen.generate_evironment()
    env_dict = gen.as_dict()
    x_0 = np.array(env["starting_positions"])
    robots = system_solve.extract_drone_positions(x_0)

    env_dict["starting_positions"] = [[-5,0],[5,0], [0,5]]
    env_dict["goals"] = [[-34.5/3,-20/3], [34.5/3,-20/3], [0,40/3]]
    env_dict["obstacles"] = [[0,0]]
    env_dict["obstacles_radius"] = [2]
    sly_cirles = []
    for obstacle, radius in zip(env_dict["obstacles"], env_dict["obstacles_radius"]):
        sly_cirles.append(Point(*obstacle).buffer(radius))
    env_dict["obstacle_map"] = shapely.ops.unary_union(sly_cirles)

    path_loss_models = []
    path_loss_models.append(radio.propagation.SimplifiedStochasticPathLoss(env_dict["obstacle_map"],1e-8,4,10,4,16))
    path_loss_models.append(radio.propagation.FreeSpacePathLoss(env_dict["obstacle_map"]))

    path_loss_model = path_loss_models[0]

    f = 2.4e9 # 2.4 GHz

    fig, ax = plt.subplots()

    for x in env_dict["starting_positions"]:
        plt.plot(x[0],x[1],'go')

    for center, radius in zip(env_dict["obstacles"], env_dict["obstacles_radius"]):
        circle = plt.Circle((center[0],center[1]),radius, edgecolor='k', facecolor='0.7', fill=True)
        ax.add_artist(circle)

    for i in range(len(env_dict["starting_positions"])-1):
        for j in range(i+1,len(env_dict["starting_positions"])):
            x1 = env_dict["starting_positions"][i]
            x2 = env_dict["starting_positions"][j]
            los = path_loss_model.get_line_of_sight(x1, x2)
            color = 'g' if los else 'r'
            plt.plot([x1[0], x2[0]],[x1[1], x2[1]], linestyle='-', color=color)
    
    ax.relim()
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()

    distances = np.linspace(0.1,50,num=150)
    fig2, ax2 = plt.subplots()
    d_off = -10
    for pl in path_loss_models:
        path_loss_over_distance = [10*np.log10(1/pl.path_loss(np.array([d_off,0]), np.array([d+d_off,0]), f)) for d in distances]
        ax2.plot(distances,path_loss_over_distance, label=type(pl).__name__)
    ax2.set_xlabel('d [m]')
    ax2.set_ylabel(r'$P_r/P_t$ [dB]')
    ax2.legend()

    #####################################################
    # ------------------Simulation----------------------#
    #####################################################
    
    swarm_factory = DroneSwarmFactory(config, env_dict)
    event_list = [events.TerminateAtArrival(swarm_factory.environment["goals"], config.sim["d_terminate"], config.sim["v_terminate"])]
    r, leaf_dict = swarm_factory.get_drone_swarm()
    x = np.array(env_dict['starting_positions']).reshape(-1)
    x_dot = np.zeros_like(x)
    state_0 = np.concatenate((x, x_dot), axis=None)
    swarm = system_solve.Drone_swarm(r, config, leaf_dict, state_0, events=event_list)
    solver = system_solve.System_solver(swarm, state_0, [0, 100], update_time=update_time,
                                    atol=config.sim['atol'],rtol=config.sim['rtol'], method="simple")
    print("Solving")
    sol = solver.solve()
    print(f"Solver done")
    accs = swarm.accelerations

    #
    # Plot connectivity of a single link
    #
    connectivity_fig, connectivity_ax = plt.subplots()
    connectivity_data = leaf_dict["radio_max_distance_controllers"][0].connection_log
    times = [i*update_time for i in range(len(connectivity_data))]
    connectivity_ax.plot(times,connectivity_data)
    connectivity_ax.set_xlabel("Time [s]")
    connectivity_ax.set_ylabel("Connection [Bool: 0=false, 1=true]")
    connectivity_ax.set_title("Connection status of a single link")
    #
    # Plot RSSI of a single link
    #
    rssi_fig, rssi_ax = plt.subplots()
    rssi_data = leaf_dict["radio_max_distance_controllers"][0].rssi_log
    rssi_ax.plot(times, rssi_data)
    rssi_ax.set_xlabel("Time [s]")
    rssi_ax.set_ylabel("RSSI [dB]")
    rssi_ax.set_title("RSSI of a single link")

    N = swarm_factory.N
    x_g = swarm_factory.x_g
    x_0 = swarm_factory.x_0
    x_o = swarm_factory.x_o
    r_o = swarm_factory.r_o    
    poly = swarm_factory.poly

    # Creating animate object for animation
    labels = dict(
            ga = "Goal Attractor",
            md = "Max Distance", 
            ca = "Colision Avoidance",
            fc = "Formation Control",
            oa = "Obstacle Avoidance Object",
            da = "Damper",
            po = "Polygon Obstacle",
            nn = "NN Obstacle Avoidance"
        )
    A = Animate(sol, N, x_g, x_0, x_o, r_o, poly, accs, update_time, swarm.t_log, labels)
    A.logartithmic_arrows = True

    print("Animating")
    # --------------------------------------------
    show_animation(A, sol, config.data_set['save_path'], save_animation=config.data_set['save_animation'])
    print("Animation done")


    
    plt.show()