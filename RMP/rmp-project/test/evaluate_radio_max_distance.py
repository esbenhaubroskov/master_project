if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from utils.environment_loader import EnvironmentDictLoader
    from environment_generator import EnvironmentGenerator
    from utils.environment_loader import EnvironmentTemplate
    from config import load_config#
    from plot import plot_diff_ttest
    import pandas as pd
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import numpy as np
    from evaluate import MetricsTTest, Evaluator
    from tqdm import tqdm

    project_folder = Path(__file__).resolve().parent.parent

    # Setup configuration
    config = load_config('experiments/quantitative_evaluation_max_distance/config_quantitative_max_distance.json')
    n_configs = 2
    n_environments = 1000

    # Create copy for configurations 
    # deepcopy: avoids python copying by reference)
    configs = [deepcopy(config) for _ in range(n_configs)]

    # Config with handcrafted obstacle avoidance:
    configs[0].uav_set["goal_attractor"] = True
    configs[0].uav_set["lidar_obstacle_avoidance"] = True
    configs[0].uav_set["max_distance"] = True
    configs[0].uav_set["collision_avoidance"] = True
    configs[0].uav_set["damper"] = True

    # Config with Residual lidar obstacle avoidance:
    configs[1].uav_set["goal_attractor"] = True
    configs[1].uav_set["lidar_obstacle_avoidance"] = True
    configs[1].uav_set["radio_max_distance"] = True
    configs[1].uav_set["collision_avoidance"] = True
    configs[1].uav_set["damper"] = True

    if config.env_params["use_template"]:
        path = Path(__file__).parent.parent.resolve() / 'scenarios' / config.env_params["custom_env"]
        template = EnvironmentTemplate(path, config)
        goal_links = template.goal_links
        formation_links = template.formation_links
        gen = EnvironmentGenerator(**template.kwargs)





    # Initialize environment and metrics 
    #gen = EnvironmentGenerator(**config.env_params)
    print("Generating environments")
    environments = []
    for _ in tqdm(range(n_environments)):
        gen.clear_obstacles()
        environments.append(gen.generate_environment(reset=False))
    #environments = [gen.generate_environment() for _ in tqdm(range(n_environments))]
    metrics_list = ["reaching_goal_metric","time_metric","distance_metric", "colliding_metric", "avg_velocity_metric", "avg_acceleration_metric"] 

    # Evaluate configuration(s)
    tests = [Evaluator(environments, configs[i], metrics_list) for i in range(n_configs)]
    results = []
    for i in range(n_configs):
        print(f"Evaluating config {i}")
        results.append(tests[i].evaluate())
    
    avg_metrics = [test.get_average_metrics() for test in tests]
    avg_metrics_df = pd.DataFrame([res.values() for res in avg_metrics],
                                index=[f"Config {i}" for i in range(n_configs)],
                                columns=avg_metrics[0].keys())
    print(avg_metrics_df)
    avg_metrics_df.to_markdown(project_folder / "experiments/quantitative_evaluation_max_distance/avg_metrics.md")
    print("Done")
    # --------- END OF SCRIPT! --------- #
    exit()