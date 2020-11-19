import gym
from ddpg import ddpgAgent

env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
args = {
    'state_dim':state_dim,
    'action_dim':action_dim,
    'hidden_dim':128,
    'GAMMA': 0.90,
    'lr_a': 0.001,
    'lr_c': 0.002,
    'tau': 0.01,
    'capacity': 10000,
    'batch_size': 64,
}

agent = ddpgAgent(**args)

for episode in range(100):
    state = env.reset()
    episode_reward = 0

    for step in range(500):
        #env.render()
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.perceive(state,action,reward,next_state,done)
        episode_reward += reward
        state = next_state

    print(f'\repisode:{episode},reward:{episode_reward}',end="")