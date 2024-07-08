from screenread import images_are_different
from screenread import getscoreimg
from screenread import getss
import cv2
import numpy as np
import pyautogui
import time
from ppoagent import PPOagent as agent
import tkinter as tk
from tkinter import filedialog

numeps = 1000
score = 0
prev_image= getscoreimg("LDPlayer")

def getstate():
    global prev_image
    global score
    img = getscoreimg("LDPlayer")
    if images_are_different(img, prev_image):
        score+=1
    if not(images_are_different(img, cv2.imread("done_morning.png", cv2.IMREAD_GRAYSCALE))):
        done = 1
    if not(images_are_different(img, cv2.imread("done_evening.png", cv2.IMREAD_GRAYSCALE))):
        done = 1
    else:
        done = 0
    prev_image = img
    ss = getss("LDPlayer")
    ss = ss/255.0
    return ss, done, score
    

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
ppo_agent = agent(statd = statd, actd = 2, epsilon=0.0001)

a = input("\n Would you like to load models? (y/n)")
if(a == 'y' or a =='Y'):
    print("Please select the actor model")
    actor_model_path = filedialog.askopenfilename(filetypes=[("keras files", "*.keras")])
    print("Please select the critic model")
    critic_model_path = filedialog.askopenfilename(filetypes=[("keras files", "*.keras")])
    
    ppo_agent.load_models(actor_model_path, critic_model_path)

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

    prev_image= getscoreimg("LDPlayer")
    

    while True:
        state, done, score = getstate()
        action = ppo_agent.selectAction(state)
        perform(action)
        state_, done, score = getstate()
        reward = score
        print(reward)

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
    if(i%5 == 0 and i!=0):
        ppo_agent.save_models()
    ppo_agent.epsilon_decay()
    reset()
