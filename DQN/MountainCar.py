import gym
import  numpy as np
import time
from collections import deque
from dqn import dqnAgent

env = gym.make('MountainCar-v0')
env.seed(17)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
args = {'state_dim':state_dim,
        'hidden_dim':128,
        'action_dim':action_dim,
        'capacity':20000,
        'batch_size':256,
        'update':200,
        'lr': 5e-3, 'EPS':1, 'EPS_COEFF':0.95, 'GAMMA':0.97
        }
#

def Transform(state):
    state = (np.array(state) + np.array((1.2, 0.0))) / np.array((1.8, 0.07))
    result = []
    result.extend(state)
    return np.array(state)

def main():
    # reward_all = [None] * EPISODE
    reward_deque = deque(maxlen=100)
    test_deque = deque(maxlen=10)
    GAMMA = 0.97
    EPISODE = 3000
    TEST = 50
    #best_model = QNetwork(2, 3)
    #best_min_reward = -300
    Agent = dqnAgent(**args)
    for episode in range(1, EPISODE + 1):
        state = Transform(env.reset())
        total_reward = 0
        done = False
        while not done:
            action = Agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = Transform(next_state)
            total_reward += reward
            #redifine reward
            reward += 100 * (GAMMA * abs(next_state[1]) - abs(state[1]))
            Agent.perceive(state, action, reward, next_state, done)
            state = next_state
        reward_deque.append(total_reward)
        #reward_all[episode - 1] = total_reward
        print(f'\repisode {episode}, total_reward = {total_reward}', end="")
        if episode % 100 == 0:
            mean_reward = np.mean(reward_deque)
            max_reward = np.max(reward_deque)
            min_reward = np.min(reward_deque)
            print(f'\repisode {episode}, min = {min_reward}, max = {max_reward}, mean = {mean_reward}')

            for _ in range(TEST):
                state = Transform(env.reset())
                test_total_reward = 0
                done = False
                while not done:
                    # env.render()
                    action = Agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    state = Transform(state)
                    test_total_reward += reward
                test_deque.append(test_total_reward)
            test_mean_reward = np.mean(test_deque)
            test_min_reward = np.min(test_deque)
            test_max_reward = np.max(test_deque)
          #  if test_min_reward > best_min_reward:
               # best_model.load_state_dict(agent.Q_network.state_dict())
               # best_min_reward = test_min_reward

            print(f'\rTEST {episode}, min = {test_min_reward}, max = {test_max_reward}, mean = {test_mean_reward}')

    #torch.save(best_model.state_dict(), "DQN_MountainCar_Torch.pth")

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)