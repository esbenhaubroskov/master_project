from email.policy import Policy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch
import numpy as np
#import config
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#DEVICE = config.learning["device"]
DEVICE = "cpu"
class PolicyNetwork(nn.Module):
    
    #Takes in observations and outputs actions mu and sigma
    def __init__(self, observation_space):
        super(PolicyNetwork, self).__init__()
        # Input: x_dot1, x_dot2, o1,o2,..,on
        # Output: f1,f2,L11,L21,L22
        # TODO: Do cholesky on M=LL^T
        self.input_layer = nn.Linear(observation_space, 128, device=DEVICE)
        self.output_layer = nn.Linear(128, 10, device=DEVICE)
        self.stack = nn.Sequential(
            nn.Linear(observation_space, 128, bias=True, device=DEVICE),
            nn.Tanh(),
            nn.Linear(128, 10, bias=True, device=DEVICE),
        )
        
    
    #forward pass
    def forward(self, x):
        #input states
        #x = self.input_layer(x)
        
        #x = F.relu(x)
        
        #actions
        #action_parameters = self.output_layer(x)
        action_parameters = self.stack(x)
        
        return action_parameters

def select_action(network, state, deterministic=False):
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
    state_tensor.required_grad = True
    
    #forward pass through network
    action_parameters = network(state_tensor)

    #get mean and std, get normal distribution

    mu, sigma = action_parameters[:, :5], torch.exp(action_parameters[:, 5:])
    if deterministic:
        action = mu.detach().reshape(-1)
        log_action = torch.ones_like(action)
    else:
        m = Normal(mu[0, :], sigma[0, :])
        #sample action, get log probability
        action = m.sample()
        log_action = m.log_prob(action)

    return action, log_action, mu[0, :], sigma[0, :]




class ActorCriticNetwork(nn.Module):
    
    def __init__(self, obs_space, action_space, action_std):
        '''
        Args:
        - obs_space (int): observation space
        - action_space (int): action space
        
        '''
        super(ActorCriticNetwork, self).__init__()
        self.action_space = action_space
        self.action_std = action_std

        self.actor = nn.Sequential(
                            nn.Linear(obs_space, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_space),
                            nn.Tanh())

        self.critic = nn.Sequential(
                        nn.Linear(obs_space, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1))
        
    def forward(self):
        ''' Not implemented since we call the individual actor and critc networks for forward pass
        '''
        raise NotImplementedError
        
    def select_action(self, state):
        ''' Selects an action given current state
        Args:
        - network (Torch NN): network to process state
        - state (Array): Array of action space in an environment

        Return:
        - (int): action that is selected
        - (float): log probability of selecting that action given state and network
        '''
    
        #convert state to float tensor, add 1 dimension, allocate tensor on device
        state = torch.from_numpy(state).float().unsqueeze(0)

        #use network to predict action probabilities
        actions = self.actor(state)

        #sample an action using the Gaussian distribution
        m = Normal(actions, 0.1)
        actions = m.sample()

        #return action
        return actions.detach().numpy().squeeze(0), m.log_prob(actions)
    
    def evaluate_action(self, states, actions):
        ''' Get log probability and entropy of an action taken in given state
        Args:
        - states (Array): array of states to be evaluated
        - actions (Array): array of actions to be evaluated
        
        '''
        
        #convert state to float tensor, add 1 dimension, allocate tensor on device
        states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)

        #use network to predict action probabilities
        actions = self.actor(states_tensor)

        #get probability distribution
        m = Normal(actions, 0.1)

        #return log_prob and entropy
        return m.log_prob(torch.Tensor(actions)), m.entropy()
        

#Proximal Policy Optimization
class PPO_policy():
    
    def __init__(self, γ, ϵ, β, δ, c1, c2, k_epoch, obs_space, action_space, action_std, α_θ, αv):
        '''
        Args:
        - γ (float): discount factor
        - ϵ (float): soft surrogate objective constraint
        - β (float): KL (Kullback–Leibler) penalty 
        - δ (float): KL divergence adaptive target
        - c1 (float): value loss weight
        - c2 (float): entropy weight
        - k_epoch (int): number of epochs to optimize
        - obs_space (int): observation space
        - action_space (int): action space
        - α_θ (float): actor learning rate
        - αv (float): critic learning rate
        
        '''
        self.γ = γ
        self.ϵ = ϵ
        self.β = β
        self.δ = δ
        self.c1 = c1
        self.c2 = c2
        self.k_epoch = k_epoch
        self.actor_critic = ActorCriticNetwork(obs_space, action_space, action_std)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor_critic.actor.parameters(), 'lr': α_θ},
            {'params': self.actor_critic.critic.parameters(), 'lr': αv}
        ])
        
        #buffer to store current batch
        self.batch = []
        self.loss = 0
    
    def process_rewards(self, rewards, terminals):
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
        for r, done in zip(reversed(rewards), reversed(terminals)):

            #Base case: G(T) = r(T)
            #Recursive: G(t) = r(t) + G(t+1)^DISCOUNT
            total_r = r + total_r * self.γ

            #no future rewards if current step is terminal
            if done:
                total_r = r

            #add to front of G
            G.insert(0, total_r)

        #whitening rewards
        G = torch.tensor(G)
        G = (G - G.mean())/G.std()

        return G
    
    def kl_divergence(self, old_lps, new_lps):
        ''' Calculate distance between two distributions with KL divergence
        Args:
        - old_lps (Array): array of old policy log probabilities
        - new_lps (Array): array of new policy log probabilities
        '''
        
        #track kl divergence
        total = 0
        
        #sum up divergence for all actions
        for old_lp, new_lp in zip(old_lps, new_lps):
            
            #same as old_lp * log(old_prob/new_prob) cuz of log rules
            total += old_lp * (old_lp - new_lp)

        return total
    
    
    def penalty_update(self):
        ''' Update policy using surrogate objective with adaptive KL penalty
        '''
        
        #get items from current batch
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        #calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        #track divergence
        divergence = 0

        #perform k-epoch update
        for epoch in range(self.k_epoch):

            #get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)
            #same as new_prob / old_prob
            ratios = torch.exp(new_lps - torch.Tensor(old_lps))

            #compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            #get loss with adaptive kl penalty
            divergence = self.kl_divergence(old_lps, new_lps).detach()
            self.loss = -ratios * advantages + self.β * divergence

            #SGD via Adam
            self.optimizer.zero_grad()
            self.loss.mean().backward()
            self.optimizer.step()

        #update adaptive penalty
        if divergence >= 1.5 * self.δ:
            self.β *= 2
        elif divergence <= self.δ / 1.5:
            self.β /= 2
        
        #clear batch buffer
        self.batch = []
            
    def clipped_update(self):
        ''' Update policy using clipped surrogate objective
        '''
        #get items from trajectory
        states = [sample[0] for sample in self.batch]
        actions = [sample[1] for sample in self.batch]
        rewards = [sample[2] for sample in self.batch]
        old_lps = [sample[3] for sample in self.batch]
        terminals = [sample[4] for sample in self.batch]

        #calculate cumulative discounted rewards
        Gt = self.process_rewards(rewards, terminals)

        log_entropies = []
        #perform k-epoch update
        for epoch in range(self.k_epoch):

            #get ratio
            new_lps, entropies = self.actor_critic.evaluate_action(states, actions)

            log_entropies.append(torch.Tensor.numpy(entropies)[0][0])
            ratios = torch.exp(new_lps - torch.stack(old_lps).squeeze(1).detach())

            #compute advantages
            states_tensor = torch.stack([torch.from_numpy(state).float().unsqueeze(0) for state in states]).squeeze(1)
            vals = self.actor_critic.critic(states_tensor).squeeze(1).detach()
            advantages = Gt - vals

            #clip surrogate objective
            surrogate1 = torch.clamp(ratios, min=1 - self.ϵ, max=1 + self.ϵ) * advantages.unsqueeze(0).T
            surrogate2 = ratios * advantages.unsqueeze(0).T

            #loss, flip signs since this is gradient descent
            self.loss = -torch.min(surrogate1, surrogate2) + self.c1 * F.mse_loss(Gt, vals) - self.c2 * entropies

            self.optimizer.zero_grad()
            self.loss.mean().backward()
            self.optimizer.step()
        
        #clear batch buffer
        self.batch = []
        return np.mean(log_entropies)


def build_model(network_type, model_path, obs_space, action_space=1, device='cpu'):
    """
    Building model

    network: class
    model_path: str
    obs_space: int
    action_space: int
    device: str
    """
    if network_type == 'pg':
        model = PolicyNetwork(obs_space)
        if type(torch.load(model_path, map_location=device)) is dict:
                # Load check point model
                model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    elif network_type == 'ppo':
        model = ActorCriticNetwork(obs_space, action_space, 1) # TODO: CHECK ACTION_STD=1???
        if type(torch.load(model_path, map_location=device)) is dict:
                # Load check point model
                model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model
     
            