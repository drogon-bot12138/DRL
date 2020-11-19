
import  numpy as np
import  gym
import time
from pgn import pgnAgent
from collections import deque

env = gym.make("MountainCar-v0")
env = env.unwrapped
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

args = {'state_dim':observation_space,
        'hidden_dim':16,
        'action_dim':action_space,
        'discount':0.995,
        'lr': 0.02
        }


def main():
    reward_deque = deque(maxlen=100)
    test_deque = deque(maxlen=10)
    EPISODE = 2000
    TEST = 30
    agent = pgnAgent(**args)
    for episode in range(1, EPISODE + 1):
        state =env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            agent.perceive(state,action, reward)
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
                #env.render()
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

        #torch.save(best_model.state_dict(), "best.pth")

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('The total time is ', time_end - time_start)