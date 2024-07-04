from screenread import getscore
from screenread import getss
import numpy as np
import pyautogui
import time
from ppoagent import PPOagent as agent

numeps = 1000
# dqn_agent = agent(statd=6627, actd=2, alpha= 0.01, decayrate= 0.2, minepsilon=0.0001)

def getstate():
    score = getscore("LDPlayer")
    done = 1 if score == 10 else 0
    score = 0 if score == 10 else score
    # ss = getss("LDPlayer").flatten()
    ss = getss("LDPlayer")
    ss = ss/255.0
    return ss, done, score
    
    # return np.concatenate([ss, np.array([score, time.time()-start, done])], axis = -1)

def perform(action):
    if action == 0:
        pyautogui.press("A")
    else:
        pyautogui.press("D")

def reset():
    pyautogui.press("W")
    time.sleep(0.5)
    pyautogui.press("W")

# statd = getss("LDPlayer").shape
statd  = (468, 584, 1)
# print(statd)
ppo_agent = agent(statd = statd, actd = 2)

for i in range(numeps):
    # start = time.time()
    score =0
    done = 0
    reward = 0
    # ss = getss("LDPlayer").flatten()
    # state = np.concatenate([ss, np.array([score, time.time()-start, done])], axis = -1)
    state, done, score = getstate()
    # print("\n\n")
    # print(state)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    

    while True:
        state, done, score = getstate()
        action = ppo_agent.selectAction(state)
        perform(action)
        time.sleep(0.2)
        state_, done, score = getstate()
        reward = score

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(state_)
        dones.append(done)

        state = state_

        if done:
            break
    
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    next_states = np.array(next_states) 
    
    ppo_agent.update(states, next_states, actions, rewards, dones)

    print("ep"+str(i)+" done")
    if(i%50 == 0 and i!=0):
        ppo_agent.save_models()
    reset()