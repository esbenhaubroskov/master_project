from tkinter.tix import MAX
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

from network import PolicyNetwork
import numpy as np
from config import config

from tqdm import tqdm
from collections import deque
from environment import DronesCollisionAvoidanceEnv

#from memory_profiler import profile

#import gc
#gc.set_debug(gc.DEBUG_LEAK)
#print(f"Garbage collection is {'not ' if not gc.isenabled() else''}enabled.")

#device to run model on 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def select_action(network, state):
    ''' Selects an action given state
    Args:
    - network (Pytorch Model): neural network used in forward pass
    - state (Array): environment state
    
    Return:
    - action.item() (float): continuous action
    - log_action (float): log of probability density of action
    
    '''
    
    #create state tensor
    state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    state_tensor.requires_grad = True
    
    #forward pass through network
    action_parameters = network(state_tensor)

    #get mean and std, get normal distribution

    mu, sigma = action_parameters[:, :5], torch.exp(action_parameters[:, 5:])

    m = Normal(mu[0, :], sigma[0, :])

    #sample action, get log probability
    action = m.sample()
    log_action = m.log_prob(action)

    return action, log_action, mu[0, :], sigma[0, :]


def process_rewards(rewards):
    ''' Converts our rewards history into cumulative discounted rewards
    Args:
    - rewards (Array): array of rewards 
    
    Returns:
    - G (Array): array of cumulative discounted rewards
    '''
    #Calculate Gt (cumulative discounted rewards)
    G = []
    
    #track cumulative reward
    total_r = 0
    
    #iterate rewards from Gt to G0
    for r in reversed(rewards):
        
        #Base case: G(T) = r(T)
        #Recursive: G(t) = r(t) + G(t+1)*DISCOUNT
        total_r = r + total_r * config.learning['discount_factor']
        
        #add to front of G
        G.insert(0, total_r)
    
    #whitening rewards
    G = torch.tensor(G).to(DEVICE)
    # Check if the number of rewards is more than 1
    if G.size(dim=-1) > 1:
        G = (G - G.mean())/G.std()
    else: # If there is only 1 reward, an error is assumed
        G = G - G
    
    return G

if __name__ == '__main__':
    from datetime import datetime
    #@profile
    def run_main():
        now = datetime.now()
        run_id = now.strftime("y%Ym%md%dh%Hmin%Ms%S") 
        run_path = config.data_set['save_path'] / 'runs' / run_id
        assert config.learning['train']
        writer = SummaryWriter(run_path) # https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
        #Make environment
        env = DronesCollisionAvoidanceEnv(config.learning['max_steps'])
        
        #Init network
        network = PolicyNetwork(env.observation.shape[0]).to(DEVICE)

        #Init optimizer
        optimizer = optim.Adam(network.parameters(), lr=0.001)

        #track scores
        scores = []

        #track recent scores
        recent_scores = deque(maxlen=100)

        #track mu and sigma
        means = []
        stds = []

        #iterate through episodes
        for episode in tqdm(range(config.learning['num_episodes'])):
            
            #reset environment, initiable variables
            state = env.reset()
            rewards = []
            log_actions = []
            score = 0
            
            
            #generate episode
            for step in range(config.learning['max_steps']):
                #env.render()
                
                #select action, clip action to be [-1, 1]
                action, la, mu, sigma = select_action(network, state)
                #action = min(max(-1, action), 1)
                
                #track distribution parameters
                #means.append(mu)
                #stds.append(sigma)

                #execute action
                new_state, reward, done, goal_reached, _ = env.step([action])
                
                #writer.add_scalar("Reward/train", reward,step+(episode*config.learning['max_steps']))
                #track episode score
                score += reward
                
                #store reward and log probability
                rewards.append(reward)
                log_actions.append(la)
                
                #end episode
                if done:
                    break
                
                #move into new state
                state = new_state
            
            writer.add_scalar("ReachedGoal/train", goal_reached, episode)
            writer.add_scalar("Reward/train", score, episode)
            #append score
            #scores.append(score)
            #recent_scores.append(score)
            
            #check for early stopping
            #if np.array(recent_scores).mean() >= config.learning['solved_score'] and len(recent_scores) >= 100:
            #    break

            #Calculate Gt (cumulative discounted rewards)
            rewards_processed = process_rewards(rewards)
            
            #adjusting policy parameters with gradient ascent
            loss = []
            for r, la in zip(rewards_processed, log_actions):
                #we add a negative sign since network will perform gradient descent and we are doing gradient ascent with REINFORCE
                loss.append(sum(-r * la))
            writer.add_scalar("Loss/train", sum(loss), episode)
            #Backpropagation
            optimizer.zero_grad()
            sum(loss).backward()
            optimizer.step()

            if episode % 100 == 0:
                #print("Save checkpoint")
                torch.save({
                            'epoch': episode,
                            'model_state_dict': network.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': sum(loss),
                            }, run_path / "check_point_model.pt")
            
        #garbage = gc.garbage
        torch.save(network.state_dict(), run_path / "model.pt")
        writer.flush()   
        env.close()
    run_main()
    