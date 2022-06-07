#from sqlalchemy import column
from events import N_dim
from rmp import RMPRoot, RMPNode
from rmp_leaf import CollisionAvoidance, GoalAttractorUni, FormationDecentralized, CollisionAvoidanceDecentralized, Damper#, MaxDistanceDecentralized,PolygonObstacleAvoidance, NNObstacleAvoidance, RadioMaxDistance
from leaf_extension import MaxDistanceDecentralized, PointCollisionAvoidance, PolygonObstacleAvoidance, NNObstacleAvoidance, RadioMaxDistance, LidarObstacleAvoidance, NNObstacleAvoidanceResidual, NNGoalResidual
from network import PolicyNetwork, build_model
import torch
from sensor import LiDARSensor
import numpy as np
from shapely.geometry import Polygon
from stable_baselines3 import PPO
import numpy as np

from radio.propagation import get_propagation_model
class DroneSwarmFactory:
    """
    Contstructs RMP trees for a drone swarm.
    """
    def __init__(self, config, environment_dict, formation_links=None, goal_links=None) -> None:
        self.config = config
        self.environment = environment_dict 
        #self.poly = [Polygon(poly) for poly in self.environment['obstacle_polygons']]
        self.poly = []
        self.N = config.uav_set['num_uavs'] # Number of robots
        self.r = RMPRoot('root')
        self.x_g = np.array(self.environment['goals']) # Position goal
        self.x_0 = np.array(self.environment['starting_positions']) # Initial position
        self.x_o = np.array(self.environment['obstacles']) # Position of obstacles 
        self.r_o = np.array(self.environment['obstacles_radius']) # Radius of obstacles
        self.robots = [] # RMP node / parents / robots obejcts
        self.lidar_sensor = LiDARSensor(self.environment['obstacle_map'], self.config.lidar['n_rays'], self.config.lidar['max_range'])
        self.leaf_dict = {   
        "collision_controllers": [],
        "formation_controllers": [],
        "obstacle_controllers": [],
        "max_distance_controllers": [],
        "polygon_obstacle_controllers": [],
        "goal_attractor_controllers": [],
        "damper_controllers": [],
        "nn_obstacle_controllers": [],
        "radio_max_distance_controllers" : [],
        "lidar_obstacle_controllers": [],
        "nn_obstacle_residual_controllers": [],
        "nn_goal_residual_controllers": []
        }
        self.formation_control_index = [0, 1, 2] # TODO: change into something more dynamic or configurable
        self.weight_fc = config.formation_flying["weight_fc"]
        self.dd = config.formation_flying["dist"]
        self.device = config.learning["device"]
        self.formation_links = None # np.empty([[],[]])
        self.goal_links = None # np.empty([[],[]])

        self._assign_formation_links(formation_links)
        self._assign_goal_links(goal_links)

        self._construct_tree()
        
    def _assign_formation_links(self, formation_links):
        if formation_links is None:
            self.formation_links = np.ones((self.N, self.N))
        else:
            self.formation_links = formation_links
    
    def _assign_goal_links(self, goal_links):
        if goal_links is None:
            self.goal_links = [i for i in range(self.N)]
        else:
            self.goal_links = goal_links

    def _create_mappings(self,i):
        phi = lambda y, i=i: np.array([[y[2 * i, 0]], [y[2 * i + 1, 0]]])
        J = lambda y, i=i: np.concatenate((
                np.zeros((2, 2 * i)),
                np.eye(2),
                np.zeros((2, 2 * (self.N - i - 1)))), axis=1)
        J_dot = lambda y, y_dot: np.zeros((2, 2 * self.N))

        return phi, J, J_dot

    def _assign_max_distance(self):
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if not self.formation_links[i,j]:
                    continue
                md = MaxDistanceDecentralized(
                    'md_robot_' + str(i),
                    self.robots[i],
                    self.robots[j],
                    d_max = self.config.max_distance["d_max"],
                    w = self.config.max_distance["w"],
                    alpha = self.config.max_distance["alpha"],
                    beta = self.config.max_distance["beta"]
                )
                self.leaf_dict['max_distance_controllers'].append(md)
                #self.mds.append(md)
    
    def _assign_radio_max_distance(self):
        self.config.radio_model['parameters']['obstacle_map'] = self.environment['obstacle_map']
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                if not self.formation_links[i,j]:
                    continue

                model = get_propagation_model(self.config.radio_model['name'], **self.config.radio_model['parameters'], 
                                                seed=self.config.env_params["seed"])

                md = RadioMaxDistance(
                    'md_robot_' + str(i),
                    self.robots[i],
                    self.robots[j],
                    rssi_min=self.config.radio_max_distance["rssi_critical"],
                    d_max = self.config.radio_max_distance["d_max"],
                    w = self.config.radio_max_distance["w"],
                    alpha = self.config.radio_max_distance["alpha"],
                    beta = self.config.radio_max_distance["beta"],
                    model = model,
                    rssi_no_signal = self.config.radio_max_distance["rssi_no_signal"],
                    distance_threshold = self.config.radio_max_distance["distance_threshold"],
                    distance_change_coefficient = self.config.radio_max_distance["distance_change_coefficient"],
                    exponential_falloff = self.config.radio_max_distance["exponential_falloff"]
                )
                self.leaf_dict['radio_max_distance_controllers'].append(md)

    def _assign_goals(self):
        for i in range(self.N):
            if not self.goal_links[i] is None:
        #       #assign goal attractor
                ga = GoalAttractorUni(
                    'ga_robot_' + str(i),
                    self.robots[i],
                    self.x_g[self.goal_links[i]],
                    alpha = self.config.goal_attractor["alpha"], # Scalar constant
                    gain = self.config.goal_attractor["gain"],  # Scalar constant
                    eta = self.config.goal_attractor["eta"])   # Scalar constant
                self.leaf_dict['goal_attractor_controllers'].append(ga)

    def _assign_collision_avoidance(self):
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    continue
                iaca = PointCollisionAvoidance(#CollisionAvoidanceDecentralized(#
                    'ca_robot_' + str(i) + '_robot_' + str(j),
                    self.robots[i],
                    self.robots[j],
                    R=self.config.collision_avoidance["R"],
                    eta=self.config.collision_avoidance["eta"])
                self.leaf_dict['collision_controllers'].append(iaca)


    def _assign_formation_control(self):
        for i in self.formation_control_index:
            for j in self.formation_control_index:
                if i == j:
                    continue
                if not self.formation_links[i,j]:
                    continue
                fc = FormationDecentralized(
                    'fc_robot_' + str(i) + '_robot_' + str(j),
                    self.robots[i],
                    self.robots[j],
                    d=self.dd,
                    w=self.weight_fc)
                self.leaf_dict['formation_controllers'].append(fc)


    def _assign_obstacle_avoidance(self):
        for i in range(self.N):
            for j in range(len(self.x_o)):
                ca = CollisionAvoidance(f'oa_ob{j}_robot_{i}',
                                        self.robots[i],
                                        None,
                                        R=self.environment['obstacles_radius'][j],
                                        c=self.x_o[j], 
                                        epsilon=self.config.obstacle_avoidance["epsilon"])
                self.leaf_dict['obstacle_controllers'].append(ca)

            for p in range(len(self.poly)):
                poa = PolygonObstacleAvoidance(f'poa_{p}_robot{i}',
                                                self.robots[i],
                                                None,
                                                self.poly[p],
                                                d_min=self.config.poly_obst_avoid["d_min"],
                                                w=self.config.poly_obst_avoid["w"],
                                                alpha=self.config.poly_obst_avoid["alpha"],
                                                eta=self.config.poly_obst_avoid["eta"],
                                                epsilon=self.config.poly_obst_avoid["epsilon"])
                self.leaf_dict['polygon_obstacle_controllers'].append(poa)

    
    def _assign_damping(self):
        for i in range(self.N):
            da = Damper(
                'da_robot_' + str(i),
                self.robots[i],
                w=self.config.damper["w"],    
                eta=self.config.damper["eta"])   
            self.leaf_dict['damper_controllers'].append(da)

    def _assign_nn_obstacle_avoidance(self):
        if self.config.learning['train']:
            for i in range(self.N):
                nn_oa = NNObstacleAvoidance(f'nn_oa_{i}', 
                                            self.robots[i],
                                            None,
                                            nn_model=None,
                                            lidar_sensor=None,
                                            n_dim=2,
                                            residual=self.config.nn_collision_avoidance["residual"],
                                            deterministic=False)
                self.leaf_dict['nn_obstacle_controllers'].append(nn_oa)
        elif self.config.learning['train'] == False: 
            # Use a pretrain model / policy 
            x_dot_size = self.config.uav_set['num_uavs'] * N_dim
            obs_space = self.config.lidar['n_rays'] + x_dot_size
            

            #model = PolicyNetwork(n_input)
               
            ## 
            #if type(torch.load(self.config.learning['model_path'], map_location=self.device)) is dict:
            #    # Load check point model
            #    model.load_state_dict(torch.load(self.config.learning['model_path'], map_location=self.device)['model_state_dict'])
            #else:
            #    model.load_state_dict(torch.load(self.config.learning['model_path'], map_location=self.device))
            
            # Custom network:
            #model = build_model(self.config.learning['network_type'],self.config.learning['model_path'], obs_space, action_space=10)

            # Stable baseline:
            model = PPO.load(self.config.learning["model_path"])
            

            for i in range(self.N):
                nn_oa = NNObstacleAvoidance(f'nn_oa{i}', 
                                            self.robots[i],
                                            None,
                                            nn_model=model,
                                            lidar_sensor=self.lidar_sensor,
                                            n_dim=2,
                                            deterministic=self.config.nn_collision_avoidance["deterministic"])
                self.leaf_dict['nn_obstacle_controllers'].append(nn_oa)
        else:
            raise Exception("The NN obstacle avoidance can't be set up.")
        
    def _assign_lidar_obstacle_avoidance(self):
        for i in range(self.N):
            #phi, J, J_dot = self._create_mappings(i) #TODO: Correct way ?

            loa = LidarObstacleAvoidance(f'loa_robot{i}',
                                           self.robots[i],
                                           None,
                                           None,
                                           None,
                                           lidar=self.lidar_sensor,
                                           verbose=False,
                                           ca_R = self.config.lidar_obst_avoid["ca_R"],
                                           ca_epsilon = self.config.lidar_obst_avoid["ca_epsilon"],
                                           ca_alpha = self.config.lidar_obst_avoid["ca_alpha"],
                                           ca_eta = self.config.lidar_obst_avoid["ca_eta"],
                                           update_weight = self.config.lidar_obst_avoid["update_weight"]
                                           )
            self.leaf_dict['lidar_obstacle_controllers'].append(loa)

    def _assign_nn_obstacle_avoidance_residual(self):
        lidar_obst_avoid_kwargs = self.config.lidar_obst_avoid

        if self.config.learning["train"]:
            for i in range(self.N):
                noar = NNObstacleAvoidanceResidual('noar_{i}', 
                                                    self.robots[i],
                                                    nn_model=None,
                                                    lidar=self.lidar_sensor,
                                                    learning=self.config.learning["train"],
                                                    **lidar_obst_avoid_kwargs
                                                    )
                self.leaf_dict['nn_obstacle_residual_controllers'].append(noar)
        
        elif not self.config.learning["train"]:
            model = PPO.load(self.config.learning["model_path"])
            for i in range(self.N):
                noar = NNObstacleAvoidanceResidual('noar_{i}', 
                                                    self.robots[i],
                                                    nn_model=model,
                                                    lidar=self.lidar_sensor,
                                                    learning=False,
                                                    **lidar_obst_avoid_kwargs
                                                    )
                self.leaf_dict['nn_obstacle_residual_controllers'].append(noar)
            
        else:
            raise Exception("Not able to train NN obstacle avoidance residual")

    def _assign_nn_goal_residual(self):
        goal_kwargs = self.config.goal_attractor

        if self.config.learning["train"]:
            for i in range(self.N):
                if not self.goal_links[i] is None:
                    gr = NNGoalResidual(
                        'gr_robot_' + str(i),
                        self.robots[i],
                        self.x_g[self.goal_links[i]],
                        nn_model=None,
                        lidar=None,
                        learning=True,
                        **goal_kwargs
                    )
                    self.leaf_dict["nn_goal_residual_controllers"].append(gr)
        
        elif not self.config.learning["train"]:
            model = PPO.load(self.config.learning["model_path"])
            for i in range(self.N):
                if not self.goal_links[i] is None:
                    gr = NNGoalResidual(
                        'gr_robot_' + str(i),
                        self.robots[i],
                        self.x_g[self.goal_links[i]],
                        nn_model=model,
                        lidar=self.lidar_sensor,
                        learning=False,
                        **goal_kwargs
                    )
                    self.leaf_dict["nn_goal_residual_controllers"].append(gr)





    def _construct_tree(self):
        for i in range(self.N):
            phi, J, J_dot = self._create_mappings(i)
            robot = RMPNode('robot_' + str(i), self.r, phi, J, J_dot)
            self.robots.append(robot)

        if self.config.uav_set['max_distance']:
            self._assign_max_distance()
        if self.config.uav_set['goal_attractor']:
            self._assign_goals()
        if self.config.uav_set['collision_avoidance']:
            self._assign_collision_avoidance()
        if self.config.uav_set['formation_control']:
            self._assign_collision_avoidance()
        if self.config.uav_set['obstacle_avoidance']:
            self._assign_obstacle_avoidance()
        if self.config.uav_set['damper']:
            self._assign_damping()
        if self.config.uav_set['nn_obstacle_avoidance']:
            self._assign_nn_obstacle_avoidance()
        if self.config.uav_set['radio_max_distance']:
            self._assign_radio_max_distance()
        if self.config.uav_set['lidar_obstacle_avoidance']:
            self._assign_lidar_obstacle_avoidance()
        if self.config.uav_set['nn_obstacle_avoidance_residual']:
            self._assign_nn_obstacle_avoidance_residual()
        if self.config.uav_set['nn_goal_residual']:
            self._assign_nn_goal_residual()
    #def _new_environment(self):
    #    self.environment = self.gen.generate_evironment()
    #    self.x_g = np.array(self.environment['goals']) # Position goal
    #    self.x_0 = np.array(self.environment['starting_positions']) # Initial position
    #    self.x_o = np.array(self.environment['obstacles']) # Position of obstacles 
    #    self.r_o = np.array(self.environment['obstacles_radius']) # Radius of obstacles

    def get_drone_swarm(self):
        return self.r, self.leaf_dict
    
    def get_new_drone_swarm(self):
        pass


if __name__ == "__main__":
    import config
    import system_solve
    import time
    from plot import plot_acceleration, show_animation, Animate, plot_rmp_tree
    import system_solve
    import metric
    import events
    import matplotlib.pyplot as plt
    from environment_generator import EnvironmentGenerator
    print("Setting up environment")
    gen = EnvironmentGenerator(config.uav_set["num_uavs"], 
                                config.env_params["n_obstacles"], 
                                config.env_params["obstacle_type"], 
                                config.env_params["drone_init_radius"], 
                                seed=config.env_params["seed"])
    environment = gen.generate_evironment()
    swarm_factory = DroneSwarmFactory(config, environment)
    print("Environment setup done")

    x = np.array(swarm_factory.environment['starting_positions']).reshape(-1)
    x_dot = np.zeros_like(x)
    state_0 = np.concatenate((x, x_dot), axis=None)
    r, leaf_dict = swarm_factory.get_drone_swarm()

    # solve the diff eq
    update_time = 0.2
    event_list = [events.TerminateAtArrival(swarm_factory.environment["goals"])]
    swarm = system_solve.Drone_swarm(r, config, leaf_dict, state_0, events=event_list)
    solver = system_solve.System_solver(swarm, state_0, [0, 100], update_time=update_time,
                                        atol=config.sim['atol'],rtol=config.sim['rtol'], method="simple")

    print("Solving")
    tic = time.time()
    sol = solver.solve()
    toc = time.time()
    print(f"Solver done. Elapsed time: {toc-tic:.3f} s")

    accs = swarm.accelerations

    # ---------------------------------------------
    # ---------- Animation and results ------------
    # ---------------------------------------------
    child_accs = r.child_accs # Robot accelerations
    root_childs = r.children # Robot leafs can be accessed

    # Print metrics
    arrivals = metric.get_arrival(sol, environment["goals"])
    metric.print_metric(sol, arrivals)


    # Plot RMP tree
    if config.data_set['plot_rmp_tree']:
        plot_rmp_tree(r, graph=None, save_fig=config.data_set['save_rmp_tree'], save_path=config.data_set['save_path'])
        
    # Plot accelerations for every robot leafs
    labels = dict(
                ga = "Goal Attractor",
                md = "Max Distance", 
                ca = "Colision Avoidance",
                fc = "Formation Control",
                oa = "Obstacle Avoidance Object",
                da = "Damper",
                po = "Polygon Obstacle",
                nn = "NN Obstacle Avoidance"
                #oa_ob2 = "Obstacle Avoidance Object 2"
            )
    i = 0
    for a in accs[1:]:
        save_path = config.data_set['save_path'] / f"Acceleration_robot_{i}"
        plot_acceleration(a, save_fig=config.data_set['save_accelerations_plot'], save_path=save_path, labels=labels)
        i += 1
    N = swarm_factory.N
    x_g = swarm_factory.x_g
    x_0 = swarm_factory.x_0
    x_o = swarm_factory.x_o
    r_o = swarm_factory.r_o    
    poly = swarm_factory.poly
    
    # Creating animate object for animation
    A = Animate(sol, N, x_g, x_0, x_o, r_o, poly, accs, update_time, swarm.t_log, labels)
    A.logartithmic_arrows = True

    print("Animating")
    # --------------------------------------------
    show_animation(A, sol, config.data_set['save_path'], save_animation=config.data_set['save_animation'])
    print("Animation done")
    plt.show()