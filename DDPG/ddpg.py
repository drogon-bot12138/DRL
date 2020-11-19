import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple

class ReplayMemory(object):
    def __init__(self, capacity,):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.memory)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(ActorNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x*2.

class CriticNetwork(nn.Module):
    def __init__(self, state_dim,action_dim ,hidden_dim):
        super(CriticNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim+action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state,action):
        inputs = torch.cat([state,action],1)
        q = F.relu(self.linear1(inputs))
        q = F.relu(self.linear2(q))
        q = self.linear3(q)
        return q

class ddpgAgent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Actor = ActorNetwork(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.Actor_target = ActorNetwork(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.Actor_target.load_state_dict(self.Actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.Actor.parameters(),lr=self.lr_a)

        self.Critic = CriticNetwork(self.state_dim,self.action_dim,self.hidden_dim).to(self.device)
        self.Critic_target = CriticNetwork(self.state_dim,self.action_dim,self.hidden_dim).to(self.device)
        self.Critic_target.load_state_dict(self.Critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.Critic.parameters(),lr=self.lr_c)

        self.Buffer = ReplayMemory(self.capacity)


    def get_action(self, state):
        s = torch.FloatTensor(state).to(self.device)
        action = self.Actor.forward(s).detach().cuda().data.cpu().numpy()
        return action

    def perceive(self,state,action,reward,next_state,done):
        self.Buffer.push(state,action,reward,next_state,done)
        self.learn()

    def learn(self):
        if len(self.Buffer) < self.batch_size:
            return
        s, a, r, s_, _ = self.Buffer.sample(self.batch_size)

        state = torch.FloatTensor(s).to(self.device)
        action = torch.FloatTensor(a).to(self.device)
        rewards = torch.FloatTensor(r).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(s_).to(self.device)

        next_action = self.Actor_target.forward(next_state).detach().to(self.device)
        q_true = rewards + self.GAMMA *self.Critic_target.forward(next_state,next_action).detach()
        q_pred = self.Critic.forward(state,action)
        loss_func = torch.nn.MSELoss()
        q_loss = loss_func(q_pred,q_true)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(self.Critic(state,self.Actor.forward(state)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.Critic_target,self.Critic)
        self.soft_update(self.Actor_target,self.Actor)

    def soft_update(self,target_network, network):
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


