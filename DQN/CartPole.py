import gym
import time
from dqn import dqnAgent


#CartPole-v0:
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
args = {'state_dim':state_dim,
        'hidden_dim':32,
        'action_dim':action_dim,
        'capacity':5000,
        'batch_size':32,
        'update':100,
        'lr': 1e-3,'EPS':1, 'EPS_COEFF':0.95,'GAMMA':0.97
        }

'''
#CartPole-v1
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
args = {'state_dim': state_dim,
        'hidden_dim': 64,
        'action_dim': action_dim,
        'capacity': 10000,
        'batch_size': 64,
        'update': 100,
        'lr': 1e-3, 'EPS': 1, 'EPS_COEFF': 0.95, 'GAMMA': 0.97
        }
#128 40k 128 eps_800s_olved
'''


EPISODE = 5001  # Episode limitation
TEST = 50  # The number of experiment test every 100 episode
def main():
    Agent = dqnAgent(**args)
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        while not done:
            action = Agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = -1 if done else 0.1
            Agent.perceive(state, action, reward, next_state, done)
            state = next_state

        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                done = False
                while not done:
                    action = Agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
            ave_reward = total_reward / TEST
            print('episode:', episode, 'Evaluation Average Reward:', ave_reward)
            '''
            #save model
            if ave_reward == STEP:
                pth_name = "CartPole_"+str(STEP)+"_.pth"
                torch.save(Agent.Q_network.state_dict(), pth_name)
            '''

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)
