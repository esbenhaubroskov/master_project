import gym
from gym import Env
from gym.spaces import Box
import numpy as np
from rmp_tree_factory import DroneSwarmFactory
from environment_generator import EnvironmentGenerator
from sensor import LiDARSensor
import system_solve
import events
from config import config 
import shapely.geometry
import numpy as np
import matplotlib.pyplot as plt


class DronesCollisionAvoidanceEnv(Env):
    """
    OpenAI Gym environment to train an obstacle avoidance residual RMP policy.
    """

    def __init__(self,simulation_length, action_space_shape=(5,), observation_space_shape=(28,), seed=None) -> None:
        # Action the obstacle avoidance can perform (f,M). Add this with the 
        float_min = np.finfo(np.float32).min
        float_max = np.finfo(np.float32).max
        self.action_space = Box(low=float_min, high=float_max, shape=action_space_shape) 
        self.sensor_min = 0.01
        self.sensor_max = config.lidar["max_range"]
        # Observation space 
        self.observation_space = Box(low=float_min,high=float_max, shape=observation_space_shape)#Box(low=np.array([self.sensor_min]), high=np.array([self.sensor_max]))
        np.finfo(np.float32).max
        self.simulation_length = simulation_length
        self.update_time = config.sim['update_time']
        self.step_counter = 0
        self.path_points = []
        # Setup environment
        if seed == None:
            self._seed = config.env_params["seed"]

        self.temp_done = False

        self.gen = EnvironmentGenerator(config.uav_set["num_uavs"], 
                            config.env_params["n_obstacles"], 
                            config.env_params["obstacle_type"], 
                            config.env_params["drone_init_radius"], 
                            seed=self._seed)
   
        self.individual_reward_episode = dict(
            total = 0,
            r_goal = 0,
            r_collide = 0,
            r_dist = 0,
            r_time = 0,
            r_control = 0,
            r_traverse = 0)
        self.total_individual_reward_episode = dict(
                    total = [],
                    r_goal =  [],
                    r_collide = [],
                    r_dist = [],
                    r_time = [],
                    r_control = [],
                    r_traverse = [])

        self.reset()
        
        
        #environment = self.gen.generate_environment()
        #self.obstacle_map = self.gen.as_multipolygon()
        #
        #self.lidar = LiDARSensor(self.obstacle_map, config.lidar['n_rays'], self.sensor_max)
        ## Create droneswarm
        #self.swarm_factory = DroneSwarmFactory(config, environment)
        #self.r, self.leaf_dict = self.swarm_factory.get_drone_swarm()
        #
        ## Set initial state
        #x = np.array(self.swarm_factory.environment['starting_positions']).reshape(-1)
        #x_dot = np.zeros_like(x)
        #state_0 = np.concatenate((x, x_dot), axis=None)
        #self.state = state_0
        #self.collided = False
        #
        ## Set simulation length
        
        #self.step_counter = 0
        #
        #self.observation = np.empty(len(x_dot)+self.lidar.n_rays)
        
        #self.event_list = [events.TerminateAtArrival(self.swarm_factory.environment["goals"])]
        #self.swarm = system_solve.Drone_swarm(self.r, config, self.leaf_dict, state_0, events=self.event_list)
        #self.solver = system_solve.System_solver(self.swarm, state_0, [0, self.simulation_length], update_time=self.update_time,
        #                                    atol=config.sim['atol'],rtol=config.sim['rtol'], method=config.sim['integrator'])



    def seed(self,seed):
        self._seed = seed
        self.gen = EnvironmentGenerator(config.uav_set["num_uavs"], 
                            config.env_params["n_obstacles"], 
                            config.env_params["obstacle_type"], 
                            config.env_params["drone_init_radius"], 
                            seed=self._seed)
        self.reset()
    
    def step(self, action):
        goal_reached = 0
        # Apply action
        if isinstance(action,np.ndarray):
            action = action
        else:
            action = action[0].cpu().numpy()
            
        f = action[0:2]
        # M is constructed with Cholesky decomposition
        # see: "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations"
        L = np.array([action[2], 0, action[3], action[4]]).reshape(2,2)
        M = L*L.T #TODO
        if config.uav_set['nn_obstacle_avoidance_residual']:
            self.solver.system.leaf_dict['nn_obstacle_residual_controllers'][0].external_set_policy(f,L)
        elif config.uav_set['nn_goal_residual']:
            self.solver.system.leaf_dict['nn_goal_residual_controllers'][0].external_set_policy(f,L)
        else:
            self.solver.system.leaf_dict['nn_obstacle_controllers'][0].external_set_policy(f,L)
        
        # Integrate
        self.solver.step_period()

        # Check if stop condition is met
        state = self.solver.state.reshape(2,-1)
        x = state[0]
        x_dot = state[1]
        x_g = self.swarm_factory.environment["goals"]
        self.step_counter += 1
        x_point = shapely.geometry.Point(x)
        if self.step_counter == self.simulation_length:
            done = True
            
        else:
            if np.linalg.norm(x-x_g) < config.sim['d_terminate'] and np.linalg.norm(x_dot) < config.sim['v_terminate']:
                done = True
                goal_reached = 1

            elif self.obstacle_map.intersects(x_point):
                self.collided = True
                done = True
                goal_reached = -1
            else:
                done = False
                goal_reached = 0
        
        
        lidar_ranges = self.lidar.get_ranges(x, invert=True)
        # observation <- x_dot, lidar ranges
        
        

        if config.uav_set['nn_obstacle_avoidance_residual']:
            f_lidar_oa, M_lidar_oa = self.solver.system.leaf_dict['nn_obstacle_residual_controllers'][0].get_base_rmp()
            f_lidar_oa = f_lidar_oa.reshape(-1)
            M_lidar_oa = M_lidar_oa.reshape(-1)
            self.observation = np.concatenate((x_dot,f_lidar_oa,M_lidar_oa,lidar_ranges))
        elif config.uav_set['nn_goal_residual']:
            self.observation = np.concatenate((x_dot,lidar_ranges))

        else:
            self.observation = np.concatenate((x_dot,lidar_ranges))

        # Calculate rewards/penalties
        r_state = [x,x_dot,lidar_ranges]
        reward = self.reward(r_state, action, done)
        self.temp_done = done
        info = {"goal_reached":goal_reached}
        return self.observation, reward, done, info

    def reward(self, r_state, action, done):
        # Reward function inspired by:
        # "Learning Vision-based Reactive Policies for Obstacle Avoidance" by Elie Aljalbout
        r_collide = 0
        r_goal = 0
        r_dist = 0
        r_control = 0
        r_time = 0
        r_traverse = 0



        x_g = self.swarm_factory.environment["goals"]
        x = r_state[0]
        x_dot = r_state[1]
        lidar_range = r_state[2]
        #step = r_state[3]
        x_point = shapely.geometry.Point(x)
        # Check robot collides with obstacle
        self.path_points.append(x)
        if self.obstacle_map.intersects(x_point):
            r_collide = -1
            
        if done: 
            r_time = - self.step_counter

            traversed_dist = traversed_distance(self.path_points)
            r_traverse = -(traversed_dist / self.start_to_goal_distances[0])
            if r_traverse>-1: # Then not reach goal
                r_traverse=-5


        
        # Calculate distance to goal
        current_distance = np.linalg.norm(x-x_g)
        normalized_distance = current_distance / self.start_to_goal_distances
        distance_change = normalized_distance - self.old_normalized_distance
        r_dist = - np.sum(distance_change)
        self.old_normalized_distance = normalized_distance

        # Check if goal is reached
        if np.linalg.norm(x-x_g) < config.sim['d_terminate'] and np.linalg.norm(x_dot) < config.sim['v_terminate']:
            r_goal = 1
            

        # Punish if acceleration is large 
        f = action[0:2]
        L = np.array([action[2], 0, action[3], action[4]]).reshape(2,2)
        M = L*L.T 
        a = np.linalg.pinv(M) * f 
        r_control = -1 * np.linalg.norm(a)
        # r_control = -1 * np.linalg.norm(action, ord=1)

        # Punish if there is a acceleration when lidar is not detecting obstacle
        if not any(self.lidar.intersection_bools):
            r_control *= config.learning["idle_gain"]

        # Reward clearance
        #r_clearance = 0.05 * x_point.distance(self.obstacle_map) # Real minimum distance to object that sensor migth not have sensed
        #r_clearance = 0.005 * np.min(lidar_range)
        
        #r_clearance = (self.lidar.max_range - np.max(lidar_range))/self.lidar.max_range # clearance reward based on inverted lidar ranges
        

        r_collide *= config.learning["collision_gain"]
        r_goal *= config.learning["goal_gain"]
        r_dist *= config.learning["distance_gain"]
        r_control *= config.learning["control_gain"]
        r_time *= config.learning["step_gain"]
        r_traverse *= config.learning["traversed_gain"]
                
        self.individual_reward_episode["r_goal"] += r_goal
        self.individual_reward_episode["r_collide"] += r_collide
        self.individual_reward_episode["r_dist"] += r_dist
        self.individual_reward_episode["r_time"] += r_time
        self.individual_reward_episode["r_control"] += r_control
        self.individual_reward_episode["r_traverse"] += r_traverse


        reward = r_collide + r_goal + r_dist + r_time + r_control + r_traverse
        
        # Clipping the reward 
        reward = max(min(reward, config.learning['max_reward']), config.learning['min_reward'])
        self.individual_reward_episode["total"] += reward
        

        return reward

    
    def render(self):        
        fig, ax  = plt.subplots()
        for ob, r in zip(self.gen.obstacles, self.gen.obstacles_radius):
            circle = plt.Circle((ob[0], ob[1]), r, color='k', fill=False)
            ax.add_artist(circle)
        # Check if stop condition is met
        state = self.solver.state.reshape(2,-1)
        ax.plot(np.array(self.gen.starting_positions)[:,0], np.array(self.gen.starting_positions)[:,1], 'or')
        ax.plot(np.array(self.gen.goals)[:,0], np.array(self.gen.goals)[:,1], 'og')
        x = state[0]
        ax.plot(x[0],x[1],'kx')
        ax.set_xlim([-10,10])
        ax.set_ylim([-10,10])
        plt.show()

    def reset(self):
        success = False
        # Try to create new enviroment
        for attempt in range(10):
            try:
            # do thing
                environment = self.gen.generate_environment()
                success = True
                break
            except:
                print("")

        if not success:
            raise Exception('Could not create environment. Consider reducing the number of obstacles or increase the environment size!')
        
        for key, value in self.individual_reward_episode.items():
            self.total_individual_reward_episode[key].append(value)

        self.individual_reward_episode = dict(
                    total = 0,
                    r_goal = 0,
                    r_collide = 0,
                    r_dist = 0,
                    r_time = 0,
                    r_control = 0,
                    r_traverse = 0)
        
        self.obstacle_map = self.gen.as_multipolygon()
        self.swarm_factory = DroneSwarmFactory(config, environment)
        self.r, self.leaf_dict = self.swarm_factory.get_drone_swarm()
        x = np.array(self.swarm_factory.environment['starting_positions']).reshape(-1)
        x_dot = np.zeros_like(x)
        state_0 = np.concatenate((x, x_dot), axis=None)
        self.state = state_0
        self.collided = False
        self.start_to_goal_distances = [np.linalg.norm(np.array(environment["goals"][i]) - np.array(environment["starting_positions"][i])) for i in range(config.uav_set["num_uavs"])]

        self.event_list = [events.TerminateAtArrival(self.swarm_factory.environment["goals"])]
        self.swarm = system_solve.Drone_swarm(self.r, config, self.leaf_dict, state_0, events=self.event_list)
        self.solver = system_solve.System_solver(self.swarm, state_0, [0, self.simulation_length], update_time=self.update_time,
                                            atol=config.sim['atol'],rtol=config.sim['rtol'], method=config.sim["integrator"])
        self.step_counter = 0
        self.path_points = []
        self.lidar = LiDARSensor(self.obstacle_map, config.lidar['n_rays'], self.sensor_max)
        lidar_ranges = self.lidar.get_ranges(x)

        if config.uav_set['nn_obstacle_avoidance_residual']:
            f_lidar_oa, M_lidar_oa = self.solver.system.leaf_dict['nn_obstacle_residual_controllers'][0].get_base_rmp()
            f_lidar_oa = f_lidar_oa.reshape(-1)
            M_lidar_oa = M_lidar_oa.reshape(-1)
            self.observation = np.concatenate((x_dot,f_lidar_oa,M_lidar_oa,lidar_ranges))
        elif config.uav_set['nn_goal_residual']:
            self.observation = np.concatenate((x_dot,lidar_ranges))
        else:
            self.observation = np.concatenate((x_dot,lidar_ranges))
        
        #self.observation = np.concatenate((x_dot,lidar_ranges))
        
        self.old_normalized_distance = 1


        return self.observation
        
def traversed_distance(points):
    dist = 0
    for i in range(len(points)-1):
        dist += np.linalg.norm(points[i]-points[i+1])
    return dist
    
