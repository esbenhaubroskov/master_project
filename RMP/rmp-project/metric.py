import numpy as np
import pandas
import shapely.geometry
from events import get_have_arrived
N_dim = 2 # 2D/3D

def get_arrival(sol, goals, d_thres=0.2, vel_thres=np.inf, goal_links=None):
    """
    Find the time index for the arrival of each drone at their designated goals
    
    sol: SolverResult
    goals: array-like with shape (n_goals, n_dims)
    d_thres: the maximum distance for determining if the drone is at the target
    vel_thres: the minimum velocity required for the drone to be considered to have arrived at the goal. 
    """
    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    
    arrival_indexes=np.empty(n_drones)
    arrival_indexes = np.full((n_drones), np.inf)

    if goal_links is None:
        goal_links = [i for i in range(len(goals))]

    for i in range(n_states):
        state = sol.y[:,i]
        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        for j in range(n_drones):
            drone_x = x[N_dim*j: N_dim*(j+1)]
            drone_x_dot = x_dot[N_dim*j: N_dim*j + 2]
            #d = np.linalg.norm(drone_x-goal[j])
            #v = np.linalg.norm(drone_x_dot)
            arrived = get_have_arrived([drone_x], [drone_x_dot], goals, d_thres, vel_thres, [goal_links[j]])
            if all(arrived) and arrival_indexes[j] == np.inf:
                arrival_indexes[j] = int(i)
    return arrival_indexes

def get_arrival_bools(sol, goals, d_thres=0.2, vel_thres=np.inf, goal_links=None):
    pass



def _validate_arrivals(arrivals, n_states):
    arrivals_ = None
    arrivals_ = np.empty(np.shape(arrivals))
    for i in range(len(arrivals_)):
        arrivals_[i] = n_states -1 if arrivals[i] == np.inf else arrivals[i]
    return arrivals_

def print_metric(sol, arrivals):
    n_drones = int(sol.y.shape[0]/(2*N_dim))
    metric_labels = ['average velocities','travelled distances','times of arrivals','average accelerations', 'average smoothness trajectory']
    agent_labels = ['Robot_'+ str(i) for i in range(n_drones)]
    metric = get_metric(sol, arrivals)
    
    df = pandas.DataFrame(metric, columns=agent_labels, index=metric_labels)
    print(df)
    return df

def get_metric(sol, arrivals):
    metric = [avg_velocity_metric(sol,arrivals),
              distance_metric(sol,arrivals),
              time_to_goal_metric(sol, arrivals), 
              avg_acceleration_metric(sol,arrivals),
              avg_smoothness_metric(sol,arrivals)]

    return metric
def get_metric_funcs():
    metrics_dict = {
        "distance_metric":distance_metric,
        "time_metric":time_to_goal_metric,
        "colliding_metric":colliding_metric, 
        "avg_smoothness_metric":avg_smoothness_metric,
        "avg_velocity_metric":avg_velocity_metric,
        "avg_acceleration_metric":avg_acceleration_metric,
        "reaching_goal_metric":reaching_goal_metric,
    }
    
    return metrics_dict



def get_metric_radio_propagation(sol,arrivals):
    #TODO: Implement this function.
    pass

def colliding_metric(sol,obstacle_map):
    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    colliding = np.full((n_drones,), False)
    for i in range(n_drones):
        for j in range(n_states - 1):
            x = sol.y[N_dim*i:N_dim*i+N_dim, j]
            x_point = shapely.geometry.Point(x)
            if obstacle_map.intersects(x_point):
                colliding[i] = True
                break
    
    return colliding

def radio_connection_metric():
    pass

def distance_metric(sol, arrivals):
    """
    Travelled distance for each agent
    
    sol: SolverResult
    arrivals: index of each agent arrival of goal in 'sol'
    """

    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    arrivals = _validate_arrivals(arrivals, n_states)
    distances = np.zeros((n_drones,))
    for j in range(n_states-1):
        for i in range(n_drones):
            if j <= arrivals[i]:
                distances[i] +=  np.linalg.norm(sol.y[N_dim*i:N_dim*i+N_dim, j]-sol.y[N_dim*i:N_dim*i+N_dim, j+1])
    
    return distances
    

def avg_velocity_metric(sol, arrivals):
    """
    Calculate the average velocity of each drone

    sol: SolverResult
    arrivals: indexes of the drone's arrivals at their respective goals
    """
    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    avg_velocities = np.empty((n_drones,))
    arrivals = _validate_arrivals(arrivals, n_states)
    for i in range(n_states):
        state = sol.y[:,i]
        state = state.reshape(2, -1)
        #x = state[0]
        x_dot = state[1]
        for j in range(n_drones):
            if i <= arrivals[j]:
                drone_x_dot = x_dot[N_dim*j:N_dim*j+2]
                velocity = np.linalg.norm(drone_x_dot)
                avg_velocities[j] += velocity
    avg_velocities = [avg_velocities[i]/arrivals[i] for i in range(len(arrivals))]
    return avg_velocities

def time_to_goal_metric(sol, arrivals):
    """
    Calculate the time before each drone arrives at their designated goals

    sol: SolverResult
    arrival: time indexes for arrivals
    """
    #arrival_times = [sol.t[int(i)] for i in arrivals]
    
    arrival_times = []
    for i in arrivals:
        try:
            arrival_times.append(sol.t[int(i)])
        except OverflowError: # In case of not arriving
            arrival_times.append(np.nan)
    return arrival_times

def reaching_goal_metric(sol, arrivals):
    arrival_times = time_to_goal_metric(sol, arrivals)
    reaching_goal = []
    for arrival_time in arrival_times:
        if np.isnan(arrival_time):
            reaching_goal.append(False)
        else:
            reaching_goal.append(True)
    return reaching_goal


def avg_acceleration_metric(sol, arrivals):
    """
    Average acceleration of each agent

    sol: SolverResult
    """
    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    avg_accelerations = np.empty((n_drones,))
    arrivals = _validate_arrivals(arrivals, n_states)
    for i in range(n_states-1):
        state = sol.y[:,i]
        state_next = sol.y[:,i+1]
        state = state.reshape(2, -1)
        state_next = state_next.reshape(2, -1)
        #x = state[0]
        x_dot = state[1]
        x_dot_next = state_next[1]
        for j in range(n_drones):
            if i <= arrivals[j]:
                drone_x_dot = x_dot[N_dim*j:N_dim*j+2]
                drone_x_dot_next = x_dot_next[N_dim*j:N_dim*j+2]
                velocity = np.linalg.norm(drone_x_dot)
                velocity_next = np.linalg.norm(drone_x_dot_next)
                acceleration = np.abs(velocity-velocity_next)/np.abs(sol.t[i]-sol.t[i+1])
                avg_accelerations[j] += acceleration

    avg_accelerations = [avg_accelerations[i]/arrivals[i] for i in range(len(arrivals))]
    return avg_accelerations
    
def avg_smoothness_metric(sol, arrivals):
    """
    Calculates the average angle between segments in the traversed paths
    
    Based on https://mdpi-res.com/d_attachment/sensors/sensors-20-06822/article_deploy/sensors-20-06822.pdf
    """

    n_drones = int(sol.y.shape[0]/(2*N_dim))
    n_states = sol.y.shape[-1]
    avg_smoothness = np.empty((n_drones,))
    arrivals = _validate_arrivals(arrivals, n_states)

    for i in range(1, n_states-1):
        state_prev = sol.y[:,i-1]
        state_prev = state_prev.reshape(2, -1)
        state = sol.y[:,i]
        state_next = sol.y[:,i+1]
        state = state.reshape(2, -1)
        state_next = state_next.reshape(2, -1)
        x_prev = state_prev[0]
        x = state[0]
        x_next = state_next[0]
        for j in range(n_drones):
            if i <= arrivals[j]:
                segment_1 = x - x_prev
                segment_2 = x_next - x
                dot_product = np.dot(segment_1, segment_2)
                intermediate = dot_product / (np.linalg.norm(segment_1)*np.linalg.norm(segment_2))
                angle = 180 * np.arccos(intermediate)/np.pi
                angle1 = 180 - angle
                angle2 = 180 + angle
                avg_smoothness[j] += np.minimum(angle1, angle2)
            
    avg_smoothness = [avg_smoothness[i]/arrivals[i] for i in range(len(arrivals))]
    return avg_smoothness



