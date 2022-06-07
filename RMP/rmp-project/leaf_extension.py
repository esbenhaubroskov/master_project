from rmp import RMPNode, RMPRoot, RMPLeaf
import numpy as np
from numpy.linalg import norm
import shapely as sly
import shapely.ops as slyops
import radio.propagation
from network import select_action
from sensor import LiDARSensor
from rmp_leaf import CollisionAvoidance, CollisionAvoidanceDecentralized, GoalAttractorUni

class DynamicObstacle:
    def __init__(self, x=np.zeros(2), x_dot=np.zeros(2)):
        self.x = x
        self.x_dot = x_dot
    
    def set_state(self, x, x_dot):
        self.x = x
        self.x_dot = x_dot

class LidarObstacleAvoidance(RMPNode):
    """
    RMP policy that avoids obstacles using a LiDAR sensor of obstacle sensing

    name: (str) name of the policy instance
    parent: (RMPNode) parent RMP node
    psi: (callable) ignored
    J: (callable) ignored
    J_dot: (callable) ignored
    lidar: (LiDARSensor)
    residual: (bool, optional) default=false. Determines if the policy should use a neural network to optimise the output. 
    verbose: (bool, optional) ignored
    kwargs: (optional keyword arguments)

    Keyword arguments:
    ca_R: (float)
    ca_epsilon: (float)
    ca_alpha: (float)
    ca_eta: (float)
    """
    def __init__(self, name, parent, psi, J, J_dot, lidar=object, verbose=False, **kwargs):
        self.lidar = lidar#LiDARSensor(self.obstacle_map, self.n_rays, self.lidar_range)
        self.n_rays = lidar.n_rays
        self.child_store = []
        self.assignments = [True for _ in range(self.n_rays)]
        self.obstacle_points = [DynamicObstacle() for _ in range(self.n_rays)]
        self.ca_R = 0.25
        self.ca_epsilon = 1e-8,
        self.ca_alpha=0.00001,
        self.ca_eta=0
        self.update_weight = 0.5
        self._points_initialized = False

        for attribute, value in kwargs.items():
            setattr(self, attribute, value)

        N = 2 # n_dim
        psi = lambda y: y
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))

        super().__init__(name, parent, psi, J, J_dot, verbose)

        for i in range(self.n_rays):

            args = dict(
                name=f"ray_obstacle_avoidance_{i}", 
                parent=self, 
                parent_param=self.obstacle_points[i],
                R=self.ca_R,
                epsilon=self.ca_epsilon,
                alpha=self.ca_alpha,
                eta=self.ca_eta
            )
            PointCollisionAvoidance(**args)
            #self.add_child(CollisionAvoidanceDecentralized(**args))
            self.child_store.append(self.children[-1])

    
    def update_params(self):
        # Get position from lidar
        _x = self.parent.x.reshape(1,-1)
        points, intersection_bools = self.lidar.get_intersection_points(*_x,get_intersection_bools=True)

        w = self.update_weight
        for i in range(self.n_rays):
            # Update obstacle points
            if not self._points_initialized:
                w = 1
                self._points_initialized = True
            point = w*points[i].reshape(-1,1) + ((1-w)*self.obstacle_points[i].x).reshape(-1,1)
            self.obstacle_points[i].set_state(point, np.zeros((2,1)))
            # Update children
            self.child_store[i].update()
            # Assign or unassign children
            if intersection_bools[i] and not self.assignments[i]:
                self.add_child(self.child_store[i])
                self.assignments[i] == True
            elif (not intersection_bools) and self.assignments[i]:
                self.remove_child(self.child_store[i])
                self.assignments[i] == False
    
    def update(self):
        self.update_params()
        self.pushforward()
            
    def pushforward(self):
        return super().pushforward()
    
    # ------------------ ADDED TEMPORARY ----------------------------
    def get_residual(self):
        check_idx = [i for i,child in enumerate(self.parent.children) if NNObstacleAvoidance==type(child)] 
        if check_idx:
            f = self.parent.children[check_idx[0]].f_residual
            M = self.parent.children[check_idx[0]].M_residual
        return f,M

    def pullback(self):
        return super().pullback()
        # if self.residual:
        #     f_nn, M_nn= self.get_residual()
        #     if not any(elem is None for elem in [f_nn, M_nn]):
        #         self.f = self.f + f_nn
        #         self.M = self.M + M_nn
        #         # TODO: Check if M is positive semi-definite
    # -----------------------------------------------------------------

        
        
       

class PointCollisionAvoidance(CollisionAvoidanceDecentralized):
    """
    Collision avoidance with a point without a radius
    """

    def __init__(self, name, parent, parent_param, c=np.zeros(2), R=0, epsilon=1e-8, alpha=0.00001, eta=0):
        super().__init__(name, parent, parent_param, c, R, epsilon, alpha, eta)
    
    def update_params(self):
        """
        update the position of the other robot
        """
        c = self.parent_param.x
        z_dot = self.parent_param.x_dot
        R = self.R
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        N = c.size

        self.psi = lambda y: np.array(norm(y - c) - R).reshape(-1,1)
        self.J = lambda y: 1.0 / norm(y - c) * (y - c).T
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                + 1 / norm(y - c) * np.eye(N)))
                


class MaxDistanceDecentralized(RMPLeaf):
    def __init__(self, name, parent, parent_param, c=np.zeros(2), d_max=5, w=1, alpha=1, beta=1):
        """
        d_max: max distance 
        w: priority
        alpha: multiplier 
        beta: exponent multiplie
        parent_param: is an other drone assosiated to keeping max distance
        """
        assert parent_param is not None
        self.d_max = d_max
        
        #psi = lambda y: (y - c) # Mapping between global and local task spaces. / Between parent and child.
        
        if parent_param:
            psi = None
            J = None
            J_dot = None
        else:
            if c.ndim == 1:
                c = c.reshape(-1, 1)
        # y and c corrospond to (x_i,x_j) respectively. c is the position of the obstacle. - See pairwise collision avoidance 

        def RMP_func(x, x_dot):
            G = w
            #d = np.linalg.norm(c - x)
            #r_hat = (c - x) / d
            #f = r_hat * alpha * np.exp((d - d_max) * beta)
            #f = np.array(alpha * np.exp((d - d_max) * beta))
            
            f = -np.array(alpha * np.exp((x) * beta))
            f = np.minimum(np.maximum(f, - 1e5), 1e5)
            f = np.tanh(f/5)*5
            M = G
                 
            return (f, M)

            

        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)
        
    def update_params(self):
        '''
        Update position of other robots
        '''
        c = self.parent_param.x
        R = self.d_max
        if c.ndim == 1:
            c = c.reshape(-1, 1)
        
        N = c.size 

        self.psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1,1)
        self.J = lambda y: 1.0  / norm(y - c) * (y - c).T / R
        self.J_dot = lambda y, y_dot: np.array(np.dot(y_dot.T,
            (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                + 1 / norm(y - c) * np.eye(N))) / R)

class RadioMaxDistance(MaxDistanceDecentralized):
    def __init__(self, name, parent, parent_param, rssi_min=-50, c=np.zeros(2), d_max=5, w=1, alpha=1, beta=1, model=None, **kwargs):
        self.rssi_min = rssi_min
        self.d_max_log = []
        self.enable_log = True
        self.rssi = None
        self.connection_log = []
        self.rssi_log = []
        self.rssi_no_signal = -100
        self.allow_max_distance_update = True
        self.positions_initialized = False

        # Max distance estimate settings
        self.distance_threshold = 2 # placeholder value
        self.distance_change_coefficient = 0.05
        self.exponential_falloff = 0.995

        self.rssi_offset = 0

        for attribute, value in kwargs.items():
            setattr(self, attribute, value)


        if model == None:
            constant_K = 1e-6
            loss_exponent = 4
            d_0 = 10
            std_dev = 4
            obstacle_map = None
            self.rf_model = radio.propagation.SimplifiedStochasticPathLoss(obstacle_map, constant_K, loss_exponent, d_0, std_dev)
        else: 
            self.rf_model = model
        

        super().__init__(name, parent, parent_param, c, d_max, w, alpha, beta)
    
    def get_max_distance_estimate(self):
        if any(self.parent.x == None):
            return self.d_max

        

        current_distance = np.linalg.norm(self.parent.x - self.parent_param.x)
        #path_loss = self.rf_model.path_loss(self.parent.x, self.parent_param.x, 2.4e9)
        #self.rssi = -10*np.log10(path_loss)
        self.rssi = self._get_new_rssi()
        self.rssi += self.rssi_offset
        
        distance_margin = self.d_max - current_distance
        signal_margin = self.rssi - self.rssi_min

        # Rudimentary function. May be replaced
        if distance_margin < self.distance_threshold or signal_margin < 0:
            return self.d_max + self.distance_change_coefficient*signal_margin
        else:
            return self.d_max * self.exponential_falloff

    def _get_new_rssi(self):
        path_loss = self.rf_model.path_loss(self.parent.x, self.parent_param.x, 2.4e9)
        rssi = -10*np.log10(path_loss)
        return rssi

    def update_params(self):
        if self.allow_max_distance_update:
            self.d_max = max(self.get_max_distance_estimate(),1)
        else:
            self.rssi = self._get_new_rssi()
        
        if self.enable_log:
            self.d_max_log.append(self.d_max)
            self.connection_log.append(0 if self.rssi < self.rssi_no_signal else 1)
            self.rssi_log.append(self.rssi)

        if self.rssi > self.rssi_no_signal:
            super().update_params()
        elif not self.positions_initialized:
            super().update_params()
            self.positions_initialized = True
        return 

class PolygonObstacleAvoidance(RMPLeaf):
    def __init__(self, name, parent, parent_param, obstacle, d_min=1, w=1, alpha=1, eta=1, epsilon=1):
        self.name = name
        self.obstacle = obstacle
        self.d_min = d_min
        self.alpha = alpha
        self.eta = eta
        self.epsilon = epsilon

        psi = None
        J = None
        J_dot = None
        

        def RMP_func(x, x_dot):
            if x < 0:
                w = 1e10
                grad_w = 0
            else:
                w = 1.0 / x ** 4
                grad_w = -4.0 / x ** 5
            u = epsilon + np.minimum(0, x_dot) * x_dot
            g = w * u

            grad_u = 2 * np.minimum(0, x_dot)
            grad_Phi = alpha * w * grad_w
            xi = 0.5 * x_dot ** 2 * u * grad_w

            M = g + 0.5 * x_dot * w * grad_u
            M = np.minimum(np.maximum(M, - 1e5), 1e5)

            Bx_dot = eta * g * x_dot

            f = - grad_Phi - xi - Bx_dot
            f = np.minimum(np.maximum(f, - 1e10), 1e10)
           
            return (f, M)


        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)
    def update_params(self):
        """
        update the position of the nearest point on the obstacle
        """
        self_x = sly.geometry.Point(self.parent.x)
        points = slyops.nearest_points(self_x, self.obstacle)
        c = np.array(points[1].coords).reshape(-1,1)
        #z_dot = self.parent_param.x_dot
        R = self.d_min
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        N = c.size

        self.psi = lambda y: np.array(norm(y - c) / R - 1).reshape(-1,1)
        self.J = lambda y: 1.0 / norm(y - c) * (y - c).T / R
        self.J_dot = lambda y, y_dot: np.dot(
            y_dot.T,
            (-1 / norm(y - c) ** 3 * np.dot((y - c), (y - c).T)
                + 1 / norm(y - c) * np.eye(N))) / R


class NNObstacleAvoidance(RMPLeaf):
    def __init__(self, name, parent, parent_param, nn_model = None, lidar_sensor = None, n_dim=2, residual=False, deterministic=False):
        self.name = name
        self.f = np.array([])
        self.M = np.array([[],[]])
        self.nn_model = nn_model
        self.lidar = lidar_sensor
        N = n_dim
        self.L = np.array([[],[]])

        psi = lambda y: (y).reshape(-1,1)
        J = lambda y: np.eye(N)
        J_dot = lambda y, y_dot: np.zeros((N, N))
        
        def RMP_func(x,x_dot):
            if nn_model:
                x_ = x.reshape(1,-1)[0]
                x_dot_ = x_dot.reshape(1,-1)[0]
               
                lidar_ranges = self.lidar.get_ranges(x_, invert=True)
                # Tjek om dimensioner passer eller giv exception
                state = np.concatenate((x_dot_,lidar_ranges))
                #action, _, _, _ = select_action(self.nn_model,state, deterministic=deterministic)
                
                #action, _= nn_model.select_action(state)
                action, _states = nn_model.predict(state, deterministic=True)
                # Apply action
                action = np.array(action) 
                self.f = np.array(action[0:2]).reshape(-1,1)
                # M is constructed with Cholesky decomposition
                # see: "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations"
                self.L = np.array([action[2], 0, action[3], action[4]]).reshape(2,2)
                self.M = self._get_M_from_L()
                
            return (self.f,self.M)
        
        RMPLeaf.__init__(self, name, parent, parent_param, psi, J, J_dot, RMP_func)

    def _get_M_from_L(self, L=None):
        if L == None:
            L = self.L
        np.fill_diagonal(L,np.absolute(np.diag(L)))
        M = L*L.T
        return M

    def update_params(self):
        pass

    def external_set_policy(self, f, L):
        self.f = np.array(f).reshape(-1,1)
        self.L = np.array(L)


class NNObstacleAvoidanceResidual(LidarObstacleAvoidance):
    """
    An RMP leaf node that avoids obstacles using a mix of a handcrafted LiDAR-based obstacle avoidance policy and a neural network.

    name: (str) name of the policy instance
    parent: (RMPNode) parent RMP node
    nn_model: ()
    lidar: (LiDARSensor)
    residual: (bool, optional) default=false. Determines if the policy should use a neural network to optimise the output. 
    verbose: (bool, optional) ignored
    learning: (bool, optional)
    kwargs: (optional keyword arguments)

    Optional keyword arguments:
    ca_R: (float)
    ca_epsilon: (float)
    ca_alpha: (float)
    ca_eta: (float)"""
    
    def __init__(self, name, parent, nn_model, lidar=object, verbose=False, learning=False, **kwargs):
        self.nn_model = nn_model
        self.f_nn = None
        self.M_nn = None
        self.L_nn = None
        self.learning = learning
        self.lidar = lidar
        self.epsilon = 1e-5
        super().__init__(name, parent, None, None, None, lidar, verbose, **kwargs)
            
    def get_base_rmp(self):
        super().pullback()
        return self.f, self.M
            
    def get_action(self, x, x_dot, f_base, M_base):
        if self.learning:
            f, L = self.f_nn, self.L_nn
        
        if not self.learning and self.nn_model:
            x_ = x.reshape(1,-1)[0]
            x_dot_ = x_dot.reshape(1,-1)[0]
            
            lidar_ranges = self.lidar.get_ranges(x_, invert=True)
            f_base = f_base.reshape(1,-1)[0]
            M_base = M_base.reshape(1,-1)[0]
            state = np.concatenate((x_dot_,f_base,M_base,lidar_ranges))
          
            action, _states = self.nn_model.predict(state, deterministic=True)
            # Apply action
            action = np.array(action) 
            f = np.array(action[0:2]).reshape(-1,1)
            # M is constructed with Cholesky decomposition
            # see: "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations"
            L = np.array([action[2], 0, action[3], action[4]]).reshape(2,2)

            #M = L*L.T
        
        return f, L
        
    def pullback(self):
        """
        Combine the handcrafted obstacle avoidance with the output of the neural network-based residual policy
        """
        f_base, M_base = self.get_base_rmp()
        f_residual, L_residual = self.get_action(self.x,self.x_dot,f_base,M_base)
        f_residual, L_residual = np.array(f_residual), np.array(L_residual)
        if None in f_residual.reshape(-1) or None in L_residual.reshape(-1):
            self.f = f_base
            self.M = M_base
        else:
            L_base = np.linalg.cholesky(M_base + (np.eye(2) * self.epsilon) + self.epsilon)
            L_total = (L_residual + L_base)
            np.fill_diagonal(L_total,np.absolute(np.diag(L_total)))
            self.M = L_total*L_total.T 
            self.f = f_residual + f_base

            #self.f = f_base + f_residual 
            #self.M = M_base + M_residual

    def external_set_policy(self, f, L):
        self.f_nn = np.array(f).reshape(-1,1)
        self.L_nn = np.array(L)

class NNGoalResidual(GoalAttractorUni):
    """
    A combined goal attraction and neural network based obstacle avoidance policy.
    The obstacle avoidance policy output is added directly to the goal attraction policy.

    parameters:
    name: (string)
    parent: (RMPNode) the parent node
    nn_model: () learned neural network model
    y_g: (ArrayLike) coordinates of the goal
    learning: (Bool) True if the model is being trained, False if the model is being deployed

    """

    def __init__(self, name, parent, y_g, nn_model, lidar, w_u=10, w_l=1, sigma=1, learning=False, **kwargs):
        self.nn_model = nn_model
        self.f_nn = None
        self.M_nn = None
        self.L_nn = None
        self.lidar = lidar
        self.learning = learning
        self.epsilon = 1e-5
        super().__init__(name, parent, y_g, w_u, w_l, sigma, **kwargs)
    
    def get_action(self, x, x_dot):
        if self.learning:
            f, L = self.f_nn, self.L_nn
        
        if not self.learning and self.nn_model:
            x_ = x.reshape(1,-1)[0]
            x_dot_ = x_dot.reshape(1,-1)[0]
            
            lidar_ranges = self.lidar.get_ranges(x_, invert=True)
            state = np.concatenate((x_dot_, lidar_ranges))
          
            action, _states = self.nn_model.predict(state, deterministic=True)
            # Apply action
            action = np.array(action) 
            f = np.array(action[0:2]).reshape(-1,1)
            # M is constructed with Cholesky decomposition
            # see: "Learning Reactive Motion Policies in Multiple Task Spaces from Human Demonstrations"
            L = np.array([action[2], 0, action[3], action[4]]).reshape(2,2)
        return f, L

    def get_base_rmp(self):
        super().pullback()
        return self.f, self.M

    def pullback(self):
        """
        Combine the handcrafted obstacle avoidance with the output of the neural network-based residual policy
        """
        f_base, M_base = self.get_base_rmp()
        f_residual, L_residual = self.get_action(self.x,self.x_dot)
        f_residual, L_residual = np.array(f_residual), np.array(L_residual)
        if None in f_residual.reshape(-1) or None in L_residual.reshape(-1):
            self.f = f_base
            self.M = M_base
        else:
            L_base = np.linalg.cholesky(M_base + (np.eye(2) * self.epsilon) + self.epsilon)
            L_total = (L_residual + L_base)
            np.fill_diagonal(L_total,np.absolute(np.diag(L_total)))
            self.M = L_total*L_total.T 
            self.f = f_residual + f_base

    def external_set_policy(self, f, L):
        self.f_nn = np.array(f).reshape(-1,1)
        self.L_nn = np.array(L)
