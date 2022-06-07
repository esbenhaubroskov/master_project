from abc import ABC, abstractmethod
from turtle import update
import shapely as sly
from shapely.geometry import LineString
import numpy as np
import scipy.stats
import random

speed_of_light = 3e8
_propagation_random = None

def free_space_path_loss(x1, x2, frequency:float) -> float:
    """
    Returns the free space path loss between two points at a given frequency

    Inputs:
    x1 : first position (array-like)
    x2 : second position (array-like)
    frequency : frequency in Hz (float)
    """
    distance = np.linalg.norm(x1 - x2)
    wave_length = speed_of_light/frequency
    fspl = (4*np.pi*distance/wave_length)**2
    return fspl

def log_normal_shadowing(mean, std_dev) -> float:
    xi = 10/np.log(10)


class PropagationModel(ABC):
    """
    Abstract base class for propagation models
    """
    propagation_random = None

    def __init__(self, obstacle_map, seed=None) -> None:
        self.obstacle_map = obstacle_map
        if PropagationModel.propagation_random is None:
            PropagationModel.propagation_random = random.Random(seed)
        super().__init__()

    @abstractmethod
    def path_loss(self, x1, x2, frequency:float) -> float:
        pass

    def get_line_of_sight(self, x1, x2):
        """
        Get line of sight between two points
        Returns : line of sight (bool)
        """
        line = LineString([x1,x2])
        line_of_sight = not line.intersects(self.obstacle_map)
        return line_of_sight
          

class FreeSpacePathLoss(PropagationModel):
    """
    Free space path loss
    """
    def __init__(self, obstacle_map) -> None:
        super().__init__(obstacle_map)
    
    
    def path_loss(self, x1, x2, frequency:float) -> float:
        """
        Path loss using Friis' equation assuming isotropic antennas.
        Returns : path loss (float)
        """
        FSPL = free_space_path_loss(x1, x2, frequency)
        return FSPL

class SimplifiedStochasticPathLoss(PropagationModel):
    """
    Combined simplified path loss and shadowing. 
    Shadowing is normal distributed.
    (Goldmsmith 2005 p. 51, see also p. 46)

    parameters:
    obstacle_map : MultiPolygon
    K : float signal strength at d_0
    loss_exponent : float
    d_0 : float reference distance
    std_dev : float shadowing standard deviation
    nlos_attenuation : float (dB) attenuation when line of sight is obstructed

    returns:
    gain : float (scalar)
    """
    def __init__(self, obstacle_map, K, loss_exponent, d_0, std_dev, nlos_attenuation, seed=None) -> None:
        self.K = K
        self.loss_exponent = loss_exponent
        self.d_0 = d_0
        self.std_dev = std_dev
        self.nlos_attenuation = nlos_attenuation
        super().__init__(obstacle_map, seed=seed)
    
    def path_loss(self, x1, x2, frequency: float) -> float:
        """
        Returns the path-loss including shadowing as a scalar (P_t/P_r)
        """
        distance = np.linalg.norm(x1 - x2)
        #gain_shadow = -scipy.stats.norm.rvs(scale=self.std_dev)
        gain_shadow = -PropagationModel.propagation_random.gauss(0, self.std_dev)
        gain_0 = 10*np.log10(self.K)
        gain_relative = -10*self.loss_exponent*np.log10(distance/self.d_0)
        gain_db = gain_0 + gain_relative + gain_shadow
        if not self.get_line_of_sight(x1,x2):
            gain_db -= self.nlos_attenuation
        gain = 10**(gain_db/10)
        return 1/gain

_models = {
    "FreeSpacePathLoss": FreeSpacePathLoss,
    "SimplifiedStochasticPathLoss": SimplifiedStochasticPathLoss
    }

def get_propagation_model(name: str, **parameters):
    return _models[name](**parameters)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    #import path
    # setting path
    directory = Path(__file__).parent.parent.resolve()
    #directory = path.path(__file__).abspath()
    sys.path.append(directory)
    print(f"path: {directory}")

    from environment_generator import EnvironmentGenerator
    from rmp_tree_factory import DroneSwarmFactory
    import system_solve
    import config

    update_time = 0.2

    gen = EnvironmentGenerator(2, 6, "circles", 15, xlim=[-20,20], ylim=[-20,20])
    env = gen.generate_evironment()
    env_dict = gen.as_dict()
    x_0 = np.array(env["starting_positions"])
    robots = system_solve.extract_drone_positions(x_0)
    
    gen.plot_environment()
