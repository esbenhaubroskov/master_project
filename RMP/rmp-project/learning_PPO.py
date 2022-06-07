import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
import numpy as np
from datetime import datetime
from pathlib import Path
import os

# -----------------------------------------
from network import PPO_policy, ActorCriticNetwork
from config import config
from utils.logger import log_git_info
from tqdm import tqdm

from environment import DronesCollisionAvoidanceEnv
# -----------------------------------------



def main():
    #Make environment
    #env = gym.make('LunarLanderContinuous-v2')
    assert config.learning['train']
    now = datetime.now()
    run_id = now.strftime("%Y-%m-%d_%H-%M-%S") 
    run_path = config.data_set['save_path'] / 'runs' / run_id
    config_path = Path(__file__).parent.resolve()  
    
    if not os.path.exists(run_path):
        os.mkdir(run_path)
    log_git_info(config_path / "config.json", run_path / "config.json")

    env = DronesCollisionAvoidanceEnv(config.learning['max_steps'])

    #seeds
    #np.random.seed(0)
    #env.seed(0)
    #torch.manual_seed(0)
    
    writer = SummaryWriter(run_path) # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html

        
    #environment parameters
    obs_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]

    #CartPole hyperparameters
    ppo_policy = PPO_policy(γ=0.99, ϵ=0.2, β=1, δ=0.01, c1=0.5, c2=0.01, k_epoch=40, 
                            obs_space=obs_space, action_space=action_space, action_std=1, α_θ = 0.0003, αv = 0.001)
    #Experiment/Policy Hyperparameters

    #number of steps to train
    TRAIN_STEPS = config.learning['num_episodes']

    #max steps per episode
    MAX_STEPS = config.learning['max_steps']

    #batch training size
    BATCH_SIZE = config.learning['batch_size']

    #solved environment score
    SOLVED_SCORE = config.learning['solved_score']
    #track scores
    scores = []

    #recent 100 scores
    recent_scores = deque(maxlen=100)

    #reset environment, initiable variables
    state = env.reset()
    curr_step = 0
    score = 0

    #run training loop
    for step in tqdm(range(1, TRAIN_STEPS)):
        
        #env.render()
        curr_step += 1

        #select action
        action, lp = ppo_policy.actor_critic.select_action(state)

        #execute action
        new_state, reward, done, goal_reached, _ = env.step(action)
        
        #track rewards
        score += reward

        #store into trajectory
        ppo_policy.batch.append([state, action, reward, lp, done])
        
        #optimize surrogate
        if step % BATCH_SIZE == 0:
            mean_entropy = ppo_policy.clipped_update()
            #ppo_policy.clipped_update()
            writer.add_scalar("Entropy/train", mean_entropy, step)
            

        #end episode
        if done or curr_step >= MAX_STEPS:
            state = env.reset()
            curr_step = 0
            scores.append(score)
            recent_scores.append(score)
            writer.add_scalar("ReachedGoal/train", goal_reached, step)
            writer.add_scalar("Reward/train", score, step)
            
            score = 0
            continue
        writer.add_scalar("Loss/train", ppo_policy.loss,step)  
          
        #check if solved environment, early stopping
        if len(recent_scores) >= 100 and np.array(recent_scores).mean() >= SOLVED_SCORE:
            break

        #move into new state
        state = new_state        
        if step % 1000 == 0:
            # print("Save checkpoint")
            torch.save({
                        'epoch': step,
                        'model_state_dict': ppo_policy.actor_critic.state_dict(),
                        'optimizer_state_dict': ppo_policy.optimizer.state_dict(),
                        'loss': ppo_policy.loss,
                        }, run_path / "check_point_model.pt")
            
        #garbage = gc.garbage
    torch.save(ppo_policy.actor_critic.state_dict(), run_path / "model.pt") # run_path / model.pt
    writer.flush()   

    #sns.set()

    #plt.plot(scores)
    #plt.ylabel('score')
    #plt.xlabel('episodes')
    #plt.title('Training score of LundarLanderContinuous with Clipped Surrogate Objective PPO')

    #reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
    #y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
    #plt.plot(y_pred)
    #plt.show()

    #evaluate policy

    #done = False
    #state = env.reset()
    #scores = []

    #for _ in tqdm_notebook(range(50)):
    #    state = env.reset()
    #    done = False
    #    score = 0
    #    while not done:
    #        #env.render()
    #        action, lp = ppo_policy.actor_critic.select_action(state)
    #        new_state, reward, done, info = env.step(env.action_space.sample())
    #        score += reward
    #        state = new_state
    #    scores.append(score)
    #env.close()
    #np.array(scores).mean()
    #env.close()
if __name__ == '__main__':
    main()
 