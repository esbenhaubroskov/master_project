from datetime import datetime
from pathlib import Path
import os
from stable_baselines3 import PPO
import torch
from tqdm import tqdm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

# -----------------------------------------
from config import config
from utils.logger import log_git_info
from environment import DronesCollisionAvoidanceEnv

class RewardWriterCallback(BaseCallback):

    def _on_training_start(self):
        self._log_freq = 2048  # log every 1000 calls

        #output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        #self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        environment = self.locals["env"].envs[0].env
        if self.n_calls % self._log_freq == 0:
        #if environment.temp_done:
            
            #self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
            #self.tb_formatter.writer.flush()
     
            environment = self.locals["env"].envs[0].env
            print(environment.total_individual_reward_episode)

            for key, value in environment.total_individual_reward_episode.items():
                avg_value = sum(value)/len(environment.total_individual_reward_episode[key])
                
                self.logger.record(f"reward_functions/{key}",avg_value)
                environment.total_individual_reward_episode[key] = []
                
  
def main(run_iteration, config, run_id):
    
    assert config.learning['train']
    num_cpu = 4  # Number of processes to use
    
    if config.learning["residual"]:
        config.learning["observation_space_shape"] = (28,) # (x_dot,f,M,O) = (2+2+4+20)
        config.learning["action_space_shape"] = (5,)
    else:
        config.learning["observation_space_shape"] = (22,) # (x_dot,O) = (2+20)
        config.learning["action_space_shape"] = (5,)

    env_kwargs = dict(simulation_length=config.learning['max_steps'], 
                    observation_space_shape=config.learning["observation_space_shape"], 
                    action_space_shape=config.learning["action_space_shape"],
                    seed=None)
    #env = make_vec_env(DronesCollisionAvoidanceEnv, n_envs=num_cpu, env_kwargs=env_kwargs)

    

    
    run_path = config.data_set['save_path'] / 'runs' / run_id /f"{run_iteration}"
    config_path = Path(__file__).parent.resolve()  
    
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    log_git_info(config_path / "config.json", run_path / f"config_id_{run_iteration}.json")

    env = DronesCollisionAvoidanceEnv(**env_kwargs)
    if config.learning["residual"]:
        policy_kwargs = dict(activation_fn=getattr(torch.nn, config.learning["activation_function"]),
                        net_arch=[dict(pi=config.learning["policy_net_arch"], vf=config.learning["value_net_arch"])])

        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log=run_path/f"PPO_stable_tensorboard_id_{run_iteration}", verbose=1)
        if config.learning["init_output_zero"]:
            model.policy.action_net.weight.data.zero_()
    else:
        model = PPO("MlpPolicy", env, tensorboard_log=run_path/f"PPO_stable_tensorboard_id_{run_iteration}", verbose=1)
    
    TIMESTEPS = 10000
    iters = 0
    while iters*TIMESTEPS < config.learning["num_episodes"]:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=RewardWriterCallback())
        model.save(run_path/f"ppo_models_id_{run_iteration}/{TIMESTEPS*iters}")

 


if __name__ == "__main__":
    reward_combinations = [
        {
        "goal_gain": 10,
        "collision_gain": 20,
        "distance_gain": 5,
        "control_gain": 0,
        "step_gain": 0,
        "idle_gain": 0,
        "traversed_gain":0},
        {
        "goal_gain": 10,
        "collision_gain": 20,
        "distance_gain": 5,
        "control_gain": 0,
        "step_gain": 0.05,
        "idle_gain": 0,
        "traversed_gain":0},
        {
        "goal_gain": 10,
        "collision_gain": 20,
        "distance_gain": 5,
        "control_gain": 0.05,
        "step_gain": 0,
        "idle_gain": 2,
        "traversed_gain":0},
        {
        "goal_gain": 10,
        "collision_gain": 20,
        "distance_gain": 5,
        "control_gain": 0.05,
        "step_gain": 0.05,
        "idle_gain": 2,
        "traversed_gain":2},
        {
        "goal_gain": 10,
        "collision_gain": 20,
        "distance_gain": 5,
        "control_gain": 0,
        "step_gain": 0,
        "idle_gain": 0,
        "traversed_gain":2},
        ]
    
    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S") 


    #for i in range(len(reward_combinations)):
    i = 3
    print(f"Training with config {i+1} of {len(reward_combinations)}")
    for key, value in reward_combinations[i].items():
        config.learning[key] = value
    main(i,config,run_id)