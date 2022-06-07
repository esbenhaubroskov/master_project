import sys 
from pathlib import Path
#sys.path.append(r"C:\Users\Asbjoern\Documents\UNI\Master_project\master-project\RMP\rmp-project")
#sys.path.append(r"C:\Users\esben\Documents\master-project\RMP\rmp-project")
sys.path.append(str(Path(__file__).resolve().parent.parent))
from rmp_tree_factory import DroneSwarmFactory

import numpy as np
import system_solve
import metric
import events
import scipy.stats 
from scipy.stats import ttest_1samp
from tqdm import tqdm

class Evaluator:
    def __init__(self, environments, config, metrics_list) -> None:
        self.environments = environments
        self.config = config
        self.metrics_list = metrics_list
        self.sol = None
        self.arrivals = None
        self.arrival_mask = None
      

    def _simulate(self,environment):
        # Return results of one simulation in one environment
        
        swarm_factory = DroneSwarmFactory(self.config, environment)
        x = np.array(swarm_factory.environment['starting_positions']).reshape(-1)
        x_dot = np.zeros_like(x)
        state_0 = np.concatenate((x, x_dot), axis=None)
        r, leaf_dict = swarm_factory.get_drone_swarm()
 
        event_list = [events.TerminateAtArrival(swarm_factory.environment["goals"], self.config.sim["d_terminate"], self.config.sim["v_terminate"]),
                        events.TerminateAtCollision(environment["obstacle_map"], self.config.uav_set["num_uavs"])]
        swarm = system_solve.Drone_swarm(r, self.config, leaf_dict, state_0, events=event_list)
        solver = system_solve.System_solver(swarm, state_0, [0, 500], update_time=self.config.sim['update_time'],
                                            atol=self.config.sim['atol'],rtol=self.config.sim['rtol'], method="simple")
        self.sol = solver.solve()

        return self.sol

    def _evaluate(self, sol, environment):
        _metric_dict = dict()
        
        metric_funcs = metric.get_metric_funcs()

        for metric_key in self.metrics_list:
            if metric_key == "colliding_metric":
                _metric_dict[metric_key] = metric_funcs[metric_key](sol,environment["obstacle_map"])
            else:
                _metric_dict[metric_key] = metric_funcs[metric_key](sol,self.arrivals)
            

        return _metric_dict


    def get_average_metrics(self):
        """
        Calculate the average of each metric. (Except for colliding metric)
        """
        eval_dict = dict()
        for key in self.metrics.keys():
            #eval_dict[key] = 0
            
            #for metric in self.metrics[key]:                
            #    eval_dict[key] += float(metric)
            #if key != "colliding_metric":
            if key != "reaching_goal_metric" and key!= "colliding_metric":
                eval_dict[key] = np.sum(self.metrics[key], where=self.arrival_mask)
                eval_dict[key] = eval_dict[key]/np.sum(self.arrival_mask)
                
            #elif key == "reaching_goal_metric":
            #    eval_dict[key] = np.sum(self.metrics[key])
            #    eval_dict[key] = eval_dict[key]/(len(self.metrics[key])*len(self.metrics[key][0]))
            else:
                eval_dict[key] = np.sum(self.metrics[key])
                eval_dict[key] = eval_dict[key]/len(self.metrics[key])
                #if eval_dict["reaching_goal_metric"]
                
        return eval_dict



        """
        eval_dict = dict()
        for key in self.metrics[0].keys():
            eval_dict[key] = 0
            for metric in self.metrics:                
                eval_dict[key] += float(metric[key][0])
            #if key != "colliding_metric":
            eval_dict[key] = eval_dict[key]/len(self.metrics)

        return eval_dict
        """
    def evaluate(self):
        self.metrics = dict()
        self.arrival_mask = []
        # Intialize metrics dict
        for key in self.metrics_list:
            self.metrics[key] = []

        for environment in tqdm(self.environments):
            sol = self._simulate(environment)
            self.arrivals = metric.get_arrival(sol, environment["goals"], self.config.sim["d_terminate"], self.config.sim["v_terminate"])
            self.arrival_mask.append(metric.reaching_goal_metric(sol,self.arrivals))

            _metric_dict = self._evaluate(sol,environment)
            for key in _metric_dict.keys():
                self.metrics[key].append(_metric_dict[key])
        #metric = self._process_evaluation(metrics)
        for key in self.metrics.keys():
            self.metrics[key] = np.array(self.metrics[key])
        return self.metrics
    
        
class MetricsTTest:

    def __init__(self, environments, config_1, config_2, metrics_list) -> None:
        self.environments = environments
        self.config_1 = config_1
        self.config_2 = config_2
        self.metrics_list = metrics_list
        self.eval_1 = Evaluator(self.environments, self.config_1, self.metrics_list)
        self.eval_2 = Evaluator(self.environments, self.config_2, self.metrics_list)
        self.scores_1 = None
        self.scores_2 = None
        self.diff_dict = dict()


    def evaluate_ttest(self):
        results = dict()
        self.scores_1 = self.eval_1.evaluate()
        self.scores_2 = self.eval_2.evaluate()
        
        for key in self.scores_1.keys():
            # Skip any metrics that are boolean
            if self.scores_1[key].dtype == bool:
                continue
            self.diff_dict[key] = self.scores_2[key] - self.scores_1[key]

        

        for key in self.diff_dict.keys():
            results[key] = ttest_1samp(self.diff_dict[key], 0, nan_policy='omit')
        
        return results

  

if __name__ == "__main__":
    from utils.environment_loader import EnvironmentDictLoader
    from environment_generator import EnvironmentGenerator
    from config import config
    import time
    from plot import plot_diff_ttest
    import pandas as pd
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import scipy
    
    config.uav_set = {
        "num_uavs": 1, 
        "max_distance": False,
        "radio_max_distance": False,
        "obstacle_avoidance": False,
        "nn_obstacle_avoidance": False,
        "collision_avoidance": False,
        "goal_attractor": True,
        "formation_control": False,
        "damper": False,
        "lidar_obstacle_avoidance":True,
        "nn_obstacle_avoidance_residual": False,
        "nn_goal_residual": False
    }

    config.env_params = {
        "n_drones":1,
        "n_obstacles": 4,
        "obstacle_type": "circles",
        "max_distance": 2.5,
        "seed": 4 
    }
    config.learning["train"] = False
    
    config1 = deepcopy(config)
    config2 = deepcopy(config)

    config2.uav_set["nn_obstacle_avoidance_residual"] = True
    config2.uav_set["lidar_obstacle_avoidance"] = False
    gen = EnvironmentGenerator(**config.env_params)
    n_environments = 20
    print("Generating environments")
    environments = [gen.generate_environment() for _ in tqdm(range(n_environments))]
    metrics_list = ["time_metric","distance_metric"]#, "colliding_metric"] #"colliding_metric",
    #test = Evaluator(environments, config1, metrics_list)
    #tic = time.time()
    #print(test.evaluate())
    #toc = time.time()
    #print(f"Tester elapsed. Elapsed time: {toc-tic:.3f} s")

    
    test = Evaluator(environments, config1, metrics_list)
    test2 = Evaluator(environments, config2, metrics_list)
    print("Evaluating config 1")
    results = test.evaluate()
    print("Evaluating config 2")
    results2 = test2.evaluate()
    #print(*np.array(results["time_metric"]).reshape(1,-1)[0], sep=', ')
    #print("\n")
    #print(*np.array(results["distance_metric"]).reshape(1,-1)[0], sep=', ')

    plt.figure(num="Time metric")
    plt.hist(np.array(results["time_metric"]).reshape(1,-1)[0], density=True, bins=50, alpha=0.5, label="Baseline")  # density=False would make counts
    plt.hist(np.array(results2["time_metric"]).reshape(1,-1)[0], density=True, bins=50, alpha=0.5, label="NN obstacle\navoidance residual")
    plt.ylabel('Probability')
    plt.xlabel('Time')
    plt.legend()

    diff = np.array(results2["time_metric"]) - np.array(results["time_metric"])

    plt.figure(num="Time metric difference")
    plt.hist(diff.reshape(1,-1)[0], density=True, bins=50, alpha=0.5)  # density=False would make counts
    plt.ylabel('Probability')
    plt.xlabel(r'$\Delta$ Time')
    plt.show()

    # plt.figure()
    # plt.hist(np.array(results["distance_metric"]).reshape(1,-1)[0], density=True, bins=50)  # density=False would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('Data')
    # plt.show()
    # ttester = MetricsTTest(environments, config1, config2, metrics_list)
    # result = ttester.evaluate_ttest()
    # print(pd.DataFrame(result, index=['t-value', 'p-value']))
    # collisions_1 = np.count_nonzero(ttester.scores_1["colliding_metric"]==True)
    # collisions_2 = np.count_nonzero(ttester.scores_2["colliding_metric"]==True)
    # print(pd.DataFrame([[collisions_1,collisions_2],
    #                     [collisions_1*100/n_environments, collisions_2*100/n_environments]], 
    #                     index=['Collisions','Percent collisions'],columns=['Baseline','Pure NN (PPO trained)']))
    # baseline = ttester.scores_1
    # diff = ttester.diff_dict
    # plt.figure()
    # plot_diff_ttest(baseline['time_metric'], diff['time_metric'], "time")
    # plt.show()
    

            
        

