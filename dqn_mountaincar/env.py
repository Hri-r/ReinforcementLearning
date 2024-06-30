import gym
import numpy as np
from test1 import dqnagent as agent
import random
import time

env = gym.make('MountainCar-v0', render_mode='human')
state_dim = env.observation_space.shape


action_dim=3

dqn_agent = agent(statd=state_dim, actd=action_dim, alpha=0.01)

num_episodes = 1000
maxre=0
for episode in range(num_episodes):
    state = env.reset()[0]
    # print(f"\n\n\n------------------------------------------\n\n\n{state} \n\n\n------------------------------------------\n\n\n")
    # print(state_dim)
    total_reward = 0
    timestart = time.time()

    while True:
        action = dqn_agent.selection(state)

        next_state, reward, done,_, _= env.step(action)
        # print(state)

        dqn_agent.store(state, action, next_state, reward, done)

        # dqn_agent.play()

        state = next_state
        total_reward += reward
        if total_reward>maxre:
            maxre=total_reward

        # if (time.time()-timestart > 30):
        #     done = True

        if done:
            env.reset()
            break

    dqn_agent.epsilon_decay()

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

print(maxre)
