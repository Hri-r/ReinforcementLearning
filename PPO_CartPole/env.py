import gym
import numpy as np
from ppoagent import PPOagent as agent
import keras
import time
import tkinter as tk
from tkinter import filedialog

env = gym.make('MountainCar-v0', render_mode='human')
state_dim = env.observation_space.shape

action_dim=env.action_space.n

ppo_agent = agent(statd=state_dim, actd=action_dim)

a = input("\n Would you like to load models? (y/n)")
if(a == 'y' or a =='Y'):
    print("Please select the actor model")
    actor_model_path = filedialog.askopenfilename(filetypes=[("keras files", "*.keras")])
    print("Please select the critic model")
    critic_model_path = filedialog.askopenfilename(filetypes=[("keras files", "*.keras")])
    
    ppo_agent.load_models(actor_model_path, critic_model_path)


num_episodes = 1000
maxre=0
for episode in range(num_episodes):
    state = env.reset()[0]
    total_reward = 0
    states, actions, rewards, next_states, dones = [], [], [], [], []
    timestart = time.time()

    while True:

        action = ppo_agent.selectAction(state)

        next_state, reward, done,_, _= env.step(action)

        if(state[1] > 0):                  # uncomment these lines if using mountaincar
            reward+= state[1]*10           # for better performance


        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)

        state = next_state
        total_reward += reward
        if total_reward>maxre:
            maxre=total_reward

        if(time.time()-timestart > 300):
            done = True

        if done:
            env.reset()
            break
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    next_states = np.array(next_states) 
    
    ppo_agent.update(states, next_states, actions, rewards, dones)

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
    if(episode%50 == 0 and episode>0):
        ppo_agent.save_models('actor.keras', 'critic.keras')

print(maxre)
