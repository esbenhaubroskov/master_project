
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
import os
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.policies import register_policy
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
import torch

# -----------------------------------------
from config import config
from utils.logger import log_git_info
from environment import DronesCollisionAvoidanceEnv


def make_env( rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DronesCollisionAvoidanceEnv(config.learning['max_steps'], seed + rank)
        return env
    set_random_seed(seed)
    return _init

def main():
    
    assert config.learning['train']
    num_cpu = 4  # Number of processes to use
    
    if config.uav_set['nn_obstacle_avoidance_residual']:
        config.learning["observation_space_shape"] = (28,) # (x_dot,f,M,O) = (2+2+4+20)
        config.learning["action_space_shape"] = (config.learning["action_space_shape"],)
    elif config.uav_set['nn_goal_residual']:
        config.learning["observation_space_shape"] = (22,) # (x_dot,O) = (2+20)
        config.learning["action_space_shape"] = (config.learning["action_space_shape"],)
    else:
        config.learning["observation_space_shape"] = (22,) # (x_dot,O) = (2+20)
        config.learning["action_space_shape"] = (config.learning["action_space_shape"],)

    env_kwargs = dict(simulation_length=config.learning['max_steps'], 
                    observation_space_shape=config.learning["observation_space_shape"], 
                    action_space_shape=config.learning["action_space_shape"],
                    seed=None)
    #env = make_vec_env(DronesCollisionAvoidanceEnv, n_envs=num_cpu, env_kwargs=env_kwargs)

    

    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S") 
    run_path = config.data_set['save_path'] / 'runs' / run_id
    config_path = Path(__file__).parent.resolve()  
    
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    log_git_info(config_path / "config.json", run_path / "config.json")

    env = DronesCollisionAvoidanceEnv(**env_kwargs)
    if config.learning["residual"]:
        policy_kwargs = dict(activation_fn=getattr(torch.nn, config.learning["activation_function"]),
                        net_arch=[dict(pi=config.learning["policy_net_arch"], vf=config.learning["value_net_arch"])])

        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log=run_path/"PPO_stable_tensorboard", verbose=1)
        if config.learning["init_output_zero"]:
            model.policy.action_net.weight.data.zero_()
    else:
        model = PPO("MlpPolicy", env, tensorboard_log=run_path/"PPO_stable_tensorboard", verbose=1)
    
    TIMESTEPS = 10000
    iters = 0
    while iters*TIMESTEPS < config.learning["num_episodes"]:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(run_path/f"ppo_models/{TIMESTEPS*iters}")


    #model.learn(total_timesteps=config.learning["num_episodes"], reset_num_timesteps=False)
    #model.save(run_path/"ppo_model")
    #del model
    #model = PPO.load(run_path/"ppo_model")

def optimize_ppo(trial):
    """ Learning hyperparamters we want to optimise"""
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 50, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        #'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        #'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        #'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }

def optimize_agent(trail):
    num_cpu = 1
    model_params = optimize_ppo(trail)
    
    config.learning["observation_space_shape"] = (28,) # (x_dot,f,M,O) = (2+2+4+20)
    config.learning["action_space_shape"] = (5,)#(config.learning["action_space_shape"],)

    env_kwargs = dict(simulation_length=config.learning['max_steps'], 
                    observation_space_shape=config.learning["observation_space_shape"], 
                    action_space_shape=config.learning["action_space_shape"],
                    seed=config.env_params["seed"])
    #env = make_vec_env(DronesCollisionAvoidanceEnv, n_envs=num_cpu, env_kwargs=env_kwargs)
    env = DronesCollisionAvoidanceEnv(**env_kwargs)
    

    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S") 
    run_path = config.data_set['save_path'] / 'runs' / run_id
    config_path = Path(__file__).parent.resolve()  
    
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    log_git_info(config_path / "config.json", run_path / "config.json")

    
    policy_kwargs = dict(activation_fn=getattr(torch.nn, config.learning["activation_function"]),
                          net_arch=[dict(pi=config.learning["policy_net_arch"], vf=config.learning["value_net_arch"])])
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **model_params, tensorboard_log=run_path/"PPO_stable_tensorboard", verbose=1)
    if config.learning["init_output_zero"]:
        model.policy.action_net.weight.data.zero_()
            #model.policy.action_net.weight.T.zero_() 
            #= #torch.zeros(torch.size([5,128]),dtype=torch.float32,device="cpu",requires_grad=True,layout=torch.strided)
    TIMESTEPS = 10000
    iters = 0
    while iters*TIMESTEPS < config.learning["num_episodes"]:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(run_path/f"ppo_model_/{TIMESTEPS*iters}")
        #model.learn(total_timesteps=config.learning["num_episodes"], reset_num_timesteps=False)
        #model.save(run_path/"ppo_model")

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return -1 * mean_reward



if __name__ == '__main__':

    optimizing = False

    if optimizing:
        study = optuna.create_study()
        study.optimize(optimize_agent, n_trials=2)
        print(study.best_params)
    else:
        main()
 