if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.environment_loader import EnvironmentDictLoader
    from environment_generator import EnvironmentGenerator
    from config import load_config#
    from plot import plot_diff_ttest
    import pandas as pd
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import numpy as np
    from evaluate import MetricsTTest, Evaluator
    from tqdm import tqdm

    # Setup configuration
    
    config = load_config(r'experiments\quantitative_evaluation_obstacle_avoidance\config_quantitative_obstacle_avoidance.json')
    n_configs = 1
    n_environments = 1000

    # Create copy for configurations 
    # deepcopy: avoids python copying by reference)
    configs = [deepcopy(config) for _ in range(n_configs)]

    # Config with circle obstacle avoidance
    configs[0].uav_set["goal_attractor"] = True
    configs[0].uav_set["obstacle_avoidance"] = True

    # Config with handcrafted obstacle avoidance:
    configs[1].uav_set["goal_attractor"] = True
    configs[1].uav_set["lidar_obstacle_avoidance"] = True
    
    
    # Config with Residual lidar obstacle avoidance:
    configs[2].uav_set["goal_attractor"] = True
    configs[2].uav_set["nn_obstacle_avoidance_residual"] = True
    configs[2].learning["model_path"] = r"experiments\quantitative_evaluation_obstacle_avoidance\residual_lidar_obstacle_avoidance_model.zip"
    
    # Config with Residual goal attractor
    configs[3].uav_set["goal_attractor"] = False
    configs[3].uav_set["nn_goal_residual"] = True
    #configs[3].learning["model_path"] = r"experiments\residual_goal_attractor\2022-05-14_10-46-53\5\ppo_models_id_5\700000.zip"
    #configs[3].learning["model_path"] = "experiments/residual_goal_attractor/2022-05-16_08-51-53/7/ppo_models_id_7/410000.zip"
    configs[3].learning["model_path"] = "experiments/residual_goal_attractor/2022-05-16_08-51-53/7/ppo_models_id_7/390000.zip"
    #configs[3].learning["model_path"] = r"experiments\quantitative_evaluation_obstacle_avoidance\residual_goal_attractor_model.zip"


    




    # Initialize environment and metrics 
    gen = EnvironmentGenerator(**config.env_params)
    print("Generating environments")
    environments = [gen.generate_environment() for _ in tqdm(range(n_environments))]
    metrics_list = ["reaching_goal_metric","colliding_metric", "time_metric","distance_metric", "avg_velocity_metric", "avg_acceleration_metric"] 

    # Evaluate configuration(s)
    tests = [Evaluator(environments, configs[i], metrics_list) for i in range(n_configs)]
    #tests =[Evaluator(environments, configs[2], metrics_list)]
    results = []
    for i in range(n_configs):
        print(f"Evaluating config {i}")
        results.append(tests[i].evaluate())
    
    avg_metrics = [test.get_average_metrics() for test in tests]
    avg_metrics_df = pd.DataFrame([res.values() for res in avg_metrics],
                                index=[f"Config {i}" for i in range(n_configs)],
                                columns=avg_metrics[0].keys())
    print(avg_metrics_df)
    avg_metrics_df.to_csv(r"experiments\quantitative_evaluation_obstacle_avoidance\avg_metrics_all.csv")
    
    print("DONE")
    # Count number of collisions
    #collisions = [np.count_nonzero(results[i]["colliding_metric"]==True) for i in range(n_configs)]
    #print(pd.DataFrame([collisions,
    #                    [collisions[i]*100/n_environments for i in range(n_configs)]], 
    #                    index=['Collisions','Percent collisions'],
    #                    columns=[f"Config {i}" for i in range(n_configs)]))



    # --------- END OF SCRIPT! --------- #
    exit()

    # Calculate Student's t-test for metrics not containing boolean
    ttester = MetricsTTest(environments, config1, config2, metrics_list)
    result = ttester.evaluate_ttest()
    print(pd.DataFrame(result, index=['t-value', 'p-value']))

    

    # Plot difference plot
    baseline = ttester.scores_1
    diff = ttester.diff_dict
    plot_diff_ttest(baseline['time_metric'], diff['time_metric'], "time")
    plt.show()