import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

#output the possibilicy of each action
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
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

class pgnAgent(object):
    def __init__(self,**kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)
            """"
            args = {'state_dim':state_dim,
                    'hidden_dim':hidden_dim,
                    'action_dim':action_dim,
                    'discount':0.95
                    'lr': 1e-1}
            """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PolicyNetwork = PolicyNetwork(self.state_dim,self.hidden_dim,self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.PolicyNetwork.parameters(), lr=self.lr)
        self.time_step = 0
        #list to store experience
        self.eps_obs = []
        self.eps_acs = []
        self.eps_rws = []

    def perceive(self,state,action, reward):
        self.eps_obs.append(state)
        self.eps_acs.append(action)
        self.eps_rws.append(reward)

    def action(self, state):
        s = torch.FloatTensor(state).to(self.device)
        out = self.PolicyNetwork.forward(s)
        with torch.no_grad():
            prob_weights = F.softmax(out, dim=0).cuda().data.cpu().numpy()
        #softmax the possibilicy ant choose and action base on the softmax output
        action = np.random.choice(range(prob_weights.shape[0]),p=prob_weights) 
        return  action

    def discount_rewards(self):
        dst_rws = []
        #计算最终目标reward
        run = 0
        for i in reversed(self.eps_rws):
            run = i + run * self.discount
            dst_rws.insert(0, run)

        dst_rws = torch.FloatTensor(dst_rws).to(self.device)
        dst_rws -= torch.mean(dst_rws)
        dst_rws /= torch.std(dst_rws)
        return dst_rws

    def learn(self):
        self.time_step += 1
        discounted_rewards = self.discount_rewards()
        softmax_input = self.PolicyNetwork.forward(torch.FloatTensor(self.eps_obs).to(self.device))
        neg_log_prob = F.cross_entropy(input=softmax_input,
                                       target=torch.LongTensor(self.eps_acs).to(self.device),
                                       reduction='none')

        loss = torch.mean(neg_log_prob * discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #clear experience
        self.eps_rws =[]
        self.eps_acs = []
        self.eps_obs = []
        
