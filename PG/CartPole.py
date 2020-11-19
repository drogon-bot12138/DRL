import numpy as np
import gym
import time
from pgn import pgnAgent
from collections import deque

'''
#CartPole-v0:
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
args = {'state_dim':state_dim,
        'hidden_dim':10,
        'action_dim':action_dim,
        'discount':0.95,
        'lr': 0.01}
'''

#CartPole-v1:
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
args = {'state_dim':state_dim,
        'hidden_dim':16,
        'action_dim':action_dim,
        'discount':0.95,
        'lr': 0.01}



def main():
    reward_deque = deque(maxlen=100)
    test_deque = deque(maxlen=10)
    EPISODE = 1000
    TEST = 30
    agent = pgnAgent(**args)
    for episode in range(1, EPISODE + 1):
        state = env.reset()
        # Train
        total_reward = 0
        done = False
        while not done:
            action = agent.action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.perceive(state, action, reward)
            state = next_state
        reward_deque.append(total_reward)
        agent.learn()

        print(f'\repisode {episode}, total_reward = {total_reward}', end="")
        if episode % 100 == 0:
            mean_reward = np.mean(reward_deque)
            max_reward = np.max(reward_deque)
            min_reward = np.min(reward_deque)
            print(f'\repisode {episode}, min = {min_reward}, max = {max_reward}, mean = {mean_reward}')

            for _ in range(TEST):
                state = env.reset()
                test_total_reward = 0
                done = False
                while not done:
                    # env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    test_total_reward += reward
                test_deque.append(test_total_reward)

            test_mean_reward = np.mean(test_deque)
            test_min_reward = np.min(test_deque)
            test_max_reward = np.max(test_deque)
            print(f'\rTEST {episode}, min = {test_min_reward}, max = {test_max_reward}, mean = {test_mean_reward}')

            #if test_mean_reward >= best_min_reward:
                #best_model = copy.deepcopy(agent.PolicyNetwork)
               # best_min_reward = test_mean_reward


                #torch.save(best_model.state_dict(), "best.pth")

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)