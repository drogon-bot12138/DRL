import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory(object):
    def __init__(self, capacity,):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            m.bias.data.zero_()

class dqnAgent(object):  # dqn Agent
    def __init__(self,**kwargs ):  # 初始化

        for key, value in kwargs.items():
            setattr(self, key, value)

            """"
            args = {'state_dim':state_dim,
                    'hidden_dim':20,
                    'action_dim':action_dim,
                    'capacity':5000,
                    'batch_size':64,
                    'update':100,
                    'lr': 1e-3,'EPS':1, 'EPS_COEFF':0.95,'GAMMA':0.97}
            """

        torch.backends.cudnn.enabled = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.Q_network = QNetwork(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_network = QNetwork(self.state_dim, self.hidden_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.Q_network.state_dict())
        self.loss_func =nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=self.lr)

        self.replay_memory = ReplayMemory(self.capacity)

        self.action_spec = int(self.action_dim)
        self.time_step = 0


    #collecting experience
    def perceive(self, state, action, reward, next_state, done):
        self.replay_memory.push(state, action, reward, next_state, done)

        if len(self.replay_memory) > self.batch_size:
            self.learn()

    #training from experience
    def learn(self):
        #update target network
        if self.time_step % self.update == 0:
            self.target_network.load_state_dict(self.Q_network.state_dict())

        self.time_step += 1

        #sampal and reshape data
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done = torch.FloatTensor(batch.done).to(self.device).unsqueeze(1)

        #compute Q
        Q = self.Q_network.forward(state_batch).gather(1, action_batch)  # 32*1
        #to compute Q_expect
        with torch.no_grad():
            argmax = self.target_network.forward(next_state_batch).detach().max(1)[1].unsqueeze(1)
            Q_next = self.target_network.forward(next_state_batch).detach().gather(1, argmax)
            Q_expect = reward_batch + (self.GAMMA * Q_next) * (1 - done)

        #Q loss_func
        loss = self.loss_func(Q_expect, Q)
        # Optimize the  network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def egreedy_action(self, state):  # epsilon-greedy

        state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)  # state_dim * 1
        Q = self.Q_network.forward(state.to(self.device)).detach()

        if random.random() <= self.EPS:
            self.EPS *= self.EPS_COEFF
            return random.randint(0, self.action_spec-1)
        else:
            self.EPS *= self.EPS_COEFF
            return torch.max(Q, 1)[1].data.to('cpu').numpy()[0]


    def action(self, state): 
        state = torch.unsqueeze(torch.FloatTensor(state).to(self.device), 0)  # state_dim * 1
        Q = self.Q_network.forward(state.to(self.device)).detach()
        return torch.max(Q, 1)[1].data.to('cpu').numpy()[0]
