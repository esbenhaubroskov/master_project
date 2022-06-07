from os import system
from scipy.integrate import RK45, RK23
from abc import ABC, abstractmethod
from rmp import RMPRoot, RMPNode
from scipy.integrate import solve_ivp
import numpy as np

N_dim = 2 # 2D or 3D

def extract_drone_positions(state, n_dim=2):
    state.reshape(2,-1)
    xs = state[0]
    positions = [xs[N_dim*i:N_dim*i+N_dim].T for i in range(int(len(xs)/N_dim))]
    return positions

class SimpleIntegrator:
    def __init__(self, dynamics, t_0, y_0, t_bound, rtol=None, atol=None, accel_max=18) -> None:
        self.dynamics = dynamics
        self.t = t_0
        self.y = y_0
        self.t_bound = t_bound
        self.status = 'running'
        self.accel_max = accel_max
    
    def step(self):
        state = self.y.reshape(2,-1)
        new_state = np.array([[],[]])
        state_dot = self.dynamics(self.t, self.y).reshape(2,-1)
        xs = state[0]
        x_dots = state_dot[0]
        x_ddots = state_dot[1]
        dt = self.t_bound - self.t
        for i in range(int(len(xs)/N_dim)):
            x = xs[N_dim*i:N_dim*i+N_dim].T
            x_dot = x_dots[N_dim*i:N_dim*i+N_dim].T
            x_ddot = x_ddots[N_dim*i:N_dim*i+N_dim].T
            accel_norm = np.linalg.norm(x_ddot)
            if self.accel_max:
                if accel_norm > self.accel_max:
                    x_ddot = (x_ddot/accel_norm)*self.accel_max
                    accel_norm = self.accel_max
                    #print("acceleration has been capped")
            #x_new = x + x_dot*dt + x_ddot*accel_norm*dt/2
            x_new = x + x_dot*dt + x_ddot*(dt**2)/2
            x_dot_new = x_dot + x_ddot*dt
            new_state_stump = np.array([x_new.T, x_dot_new.T])
            new_state = np.append(new_state, new_state_stump,1)
        
        self.t += dt
        self.y = new_state.flatten()
        self.status = 'stopped'

        # x_new = x_dot*dt + x_ddot^2*dt

_solve_methods = {"RK45": RK45,
                "RK23": RK23,
                "simple": SimpleIntegrator}        

class Dynamic_system(ABC):
    def __init__(self, initial_state, t=0, events=[]):
        self.initial_state = initial_state
        self.t = t
        self.t_log = [t]
        self.events = events
        self.terminate = False
        return
    
    @abstractmethod
    def update(self, t ,state):
        """ 
        Update the parameters of the dynamical system.
        """
        for event in self.events:
            event(self, t, state)
        self.t = t
        self.t_log.append(t)

    @abstractmethod
    def dynamics(self, t, state):
        """
        Static dynamics
        ------------------------
        :param t: time
        :param state: a state vector containing x and x_dot
        :return state_dot:
        """

class Drone_swarm(Dynamic_system):
    def __init__(self, root, config, leaf_dict, initial_state, t=0, events=[]):
        """
        Initialize object paramters and RMP tree
        ------------------------------------------
        :param root: RMPRoot of the drone swarm
        :param config: config information
        :param leaf_dict: dictionary of leaf nodes that need to be updated
        :param initial_state: the initial state is a concatenation of x and x_dot at the initiation time
        :param t: initial time
        """
        self.root = root
        self.config = config
        self.leaf_dict = leaf_dict
        
        # Initialize acceleration dict
        self.accelerations = []
        #self.accelerations = self._init_dict()
        
        # Initialize the RMP tree
        self._init_tree(initial_state)
        self._init_dict()
        super().__init__(initial_state, t, events=events)

    def _init_tree(self, state):
        """
        Set initial state of the RMP tree
        """
        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        self.root.set_root_state(x,x_dot)
        self.root.pushforward()
        self._update_leaf_policies()
        self.root.pullback()

    def _init_dict(self):
        """
        Initialize acceleration dict for robots and their policies. 
        """
        robot_dict = self.root.child_accs
        root_childs = self.root.children
        temp = dict()

        # Append Robots dict of acceleration
        for key in robot_dict.keys():
            temp[key] = []
        self.accelerations.append(temp)

        # Append leaf dict for each robot
        for rc in root_childs:
            temp = dict() 
            leaf_dict = rc.child_accs
            for key in leaf_dict.keys():
                temp[key] = []
            self.accelerations.append(temp)


    def update(self, t, state):
        """
        Update leaf nodes
        """
        super().update(t,state)
        # -----------------
        # Save child accelerations
        # -----------------
        # Append Robots dict of acceleration
        robot_dict = self.root.child_accs
        root_childs = self.root.children
        i = 0 

        for key in robot_dict.keys():
            self.accelerations[i][key].append([robot_dict[key][-1],t])

        # Append leaf dict for each robot
        for rc in root_childs:
            i += 1
            leaf_dict = rc.child_accs
            for key in leaf_dict.keys():
                if not key in self.accelerations[i]:
                    self.accelerations[i][key] = []
                self.accelerations[i][key].append([leaf_dict[key][-1],t])
    
        self._update_leaf_policies()
    

    def _update_leaf_policies(self):
        for key in self.leaf_dict.keys():
            if len(self.leaf_dict[key]) > 0:
                for leaf in self.leaf_dict[key]:
                    leaf.update()
    
    def dynamics(self, t, state):
        """
        Compute static dynamics without updating leaf nodes
        ------------------------
        :param t: time
        :param state: a state vector containing x and x_dot
        :return state_dot:
        """

        state = state.reshape(2, -1)
        x = state[0]
        x_dot = state[1]
        self.root.set_root_state(x, x_dot)
        self.root.pushforward()
        #self._update_leaf_policies()
        self.root.pullback()
        x_ddot = self.root.resolve()
        state_dot = np.concatenate((x_dot, x_ddot), axis=None)
        return state_dot

class SolverResult:
    def __init__(self, y, t):
        self.y = np.array(y).T
        self.t = t

class System_solver:

    def __init__(self, system, state, t_span, update_time=0.01, method="RK45",
                atol=1e-3, rtol=1e-6):
        """
        Initialize object parameters and create integrator object
        """
        self.system = system
        self.state = state
        self.update_time = update_time
        self.method = _solve_methods[method]
        self.t_span = t_span
        self.terminated = False
        self._period = 0

        self.y = [state]
        self.t = t_span[0]
        self.ts = [0]
        self.integrator = self.method(self.system.dynamics, 
                    t_span[0], # t_0
                    self.system.initial_state, 
                    t_span[0] + self.update_time,
                    rtol=rtol,
                    atol=atol)
        return

    def step_period(self):
        """
        Integrate for one period defined by the update_time
        """
        self.system.update(self.t, self.state)
        self.integrator.status = 'running'
        
        # Step the integrator until t_bound
        while(self.integrator.status == 'running'):
            self.integrator.step()
        
        # Update solver parameters
        self.y.append(self.integrator.y) # Append current state to y
        self.state = self.integrator.y
        self.t = self.integrator.t
        self.ts.append(self.integrator.t)
        self._period += 1

        # Set stop time for next peroid
        self.integrator.t_bound += self.update_time

        return self.integrator.y, self.integrator.t
        

    def solve(self):
        """
        Solve the system until the end of t_span
        """
        while(self.t < self.t_span[1]):
            self.step_period()
            if self.system.terminate:
                self.terminated = True
                break

        return SolverResult(self.y, self.ts)
    
    def ivp_solve(self):
        """
        Solve the system using solve_ivp()
        """
        sol = solve_ivp(self.system.dynamics, self.t_span, self.system.initial_state)
        return sol

    def get_terminated(self):
        return self.system.terminate