from rmp_leaf import CollisionAvoidance
import numpy as np
import config
import shapely

N_dim = 2 # 2D/3D

def placeholder_event(dyn_sys, t, state):
    print("hello")

class ObstacleSensor:
    def __init__(self, obstacle_center, sense_R, avoid_R):
        self.c = np.array(obstacle_center)
        self.sense_R = sense_R
        self.avoid_R = avoid_R
        if self.c.ndim == 1:
                self.c = self.c.reshape(-1, 1)
                

    def __call__(self, dyn_sys, t, state):
        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        N = int(len(x)/2)

        # for each robot
        for i in range(N):
            x_ = np.array([x[2*i:2*i+2]]).T
            robot = dyn_sys.root.children[i]
            # if robot within range of obstacle
            if (np.linalg.norm(x_ - self.c) < self.sense_R):
                preexisting = False
                for leaf in robot.children:
                    if(type(leaf)==CollisionAvoidance and (leaf.c==self.c).all()):
                        preexisting = True
                
                if not preexisting:
                    print(f"Added obstacle avoidance for {robot.name}")
                    # Add obstacle avoidance policy to robot 
                    ca = CollisionAvoidance(f'oa_ob_{robot.name}',
                                    robot,
                                    None,
                                    R=self.avoid_R,
                                    c=self.c, 
                                    epsilon=0.2)
                   
                    dyn_sys.leaf_dict["obstacle_controllers"].append(ca)
            # If robot not in obstacle range, remove obstacle avoidance policy from robot
            else:
                for leaf in robot.children:
                    if(type(leaf)==CollisionAvoidance and (leaf.c==self.c).all()):
                        robot.remove_child(leaf)
                        print(f"Removed obstacle avoidance for {robot.name}")

class TerminateAtArrival:
    """
    Set the terminate flag to stop the solver when all drones are at their goals
    """
    def __init__(self, goals, d_thres=0.2, vel_thres=np.inf, goal_links=None):
        self.goals = goals
        self.d_thres = d_thres
        self.vel_thres = vel_thres
        self.goal_links = goal_links

        if self.goal_links is None:
            self.goal_links = [i for i in range(len(self.goals))]
            
    
    def __call__(self, dyn_sys, t, state):
        n_drones = int(state.shape[0]/(2*N_dim))
        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        xs = [x[N_dim*i: N_dim*(i+1)] for i in range(n_drones)]
        x_dots = [x_dot[N_dim*i: N_dim*i + 2] for i in range(n_drones)]
        
        arrivals = get_have_arrived(xs, x_dots, self.goals, self.d_thres, self.vel_thres, self.goal_links)
        if all(arrivals):
            terminate = True
        else:
            terminate = False

        #terminate = True
        #for i in range(n_drones):
        #    drone_x = x[N_dim*i: N_dim*(i+1)]
        #    drone_x_dot = x_dot[N_dim*i: N_dim*i + 2]
        #    d = np.linalg.norm(drone_x-self.goals[i])
        #    v = np.linalg.norm(drone_x_dot)
        #    if d > self.d_thres or v > self.vel_thres:
        #        terminate = False
        dyn_sys.terminate = terminate

class TerminateAtCollision:

    def __init__(self, obstacle_map, n_drones, n_dims=2):
        self.n_dims = n_dims
        self.obstacle_map = obstacle_map
        self.n_drones = n_drones
        self.points = [shapely.geometry.Point([0,0]) for _ in range(self.n_drones)]

    def __call__(self, dyn_sys, t, state):
        n_drones = int(state.shape[0]/(2*N_dim))
        state = state.reshape(2, -1)
        x = state[0]
        xs = [x[self.n_dims*i: self.n_dims*(i+1)] for i in range(n_drones)]
        for i in range(self.n_drones):
            self.points[i].coords = xs[i] 
        
        for point in self.points:
            if self.obstacle_map.intersects(point):
                dyn_sys.terminate = True
                break




        
def get_have_arrived(positions, velocities, goals, d_thres, v_thres, goal_links=None, d_thres_unassigned=1):
    if goal_links is None:
        goal_links = [i for i in range(len(goals))]
    arrivals = []
    #np.linalg.norm(x-x_g) < config.sim['d_terminate'] and np.linalg.norm(x_dot) < config.sim['v_terminate']
    for i in range(len(positions)):
        if goal_links[i] is None:
            distances = [np.linalg.norm(positions[i]-goals[j]) for j in range(len(goals))]
            speed = np.linalg.norm(velocities[i])
            if min(distances) < d_thres_unassigned and speed < v_thres:
                arrivals.append(True)
            else:
                arrivals.append(False)
        elif np.linalg.norm(positions[i]-goals[goal_links[i]]) < d_thres and np.linalg.norm(velocities[i]) < v_thres:
            arrivals.append(True)
        else:
            arrivals.append(False)
    
    return arrivals

def get_swarm_arrival(arrivals):
    """
    Check if all drones in swarm has reached goal.
    """
    reached_goal_area = False
    if all(arrivals):
        reached_goal_area = True
    
    return reached_goal_area        