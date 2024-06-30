import tensorflow as tf
from keras._tf_keras.keras.layers import Dense
import keras._tf_keras.keras as keras
import numpy as np
from collections import deque
import random

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), name='q_network'):
        super(QNetwork, self).__init__(name=name)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_units = hidden_units

        # Create the Q-network
        self.q_network = self.build_q_network()

    def build_q_network(self):
        model = tf.keras.Sequential()

        # model.add(tf.keras.layers.InputLayer(shape = self.state_dim))
        # print(self.state_dim)
        model.add(keras.layers.InputLayer(shape=self.state_dim))

        # Hidden layers
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))

        # Output layer
        model.add(tf.keras.layers.Dense(self.action_dim, activation=None))

        return model

    def call(self, state):
        # Compute Q-values for the given state using the Q-network
        q_values = self.q_network(state)
        return q_values

# Target Q-network has the same structure as Q-network
TargetQNetwork = QNetwork

class dqnagent():
    def __init__(self, statd, actd, gamma = 0.99, alpha = 0.0005, epsilon = 1, minepsilon = 0.01, decayrate = 0.5, memcap = 10000, batchlen=64, updatefreq=100):
        self.statd = statd
        self.actd  = actd
        self.gamma  = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.minepsilon = minepsilon
        self.decayrate = decayrate

        self.optimizer= tf.keras.optimizers.Adam(alpha)

        self.qnet = QNetwork(self.statd, self.actd)
        self.tqnet = TargetQNetwork(self.statd, self.actd)
        load = input("load model?(Y/n)");
        if load =="Y":
            self.qnet=tf.keras.models.load_model(r"C:\Users\harik\OneDrive\Desktop\sac\dqn_mountaincar\qnet.tf")
            self.tqnet =tf.keras.models.load_model(r"C:\Users\harik\OneDrive\Desktop\sac\dqn_mountaincar\tarqnet.tf")

        # print(self.statd)
        # print(self.actd)
        self.replaybuffer = deque(maxlen = memcap)
        self.batchlen= batchlen
        self.updatefreq=updatefreq

        self.updates = 0

    def selection(self, state):
        if np.random.rand()<self.epsilon :
            return np.random.choice(self.actd)
             
        else:
            print(state)
            state = np.reshape(state, (1, self.statd[0]))
            qvals = self.qnet.predict(state)
            return np.argmax(qvals)
    
    def store(self, state, action, state_, reward, done):
        # print(state)
        # state=np.array([state])
        self.replaybuffer.append([state, action, state_, reward, done])
        
        # print(f"\n\n\n------------------------------------------\n\n\n{len(self.replaybuffer)} \n\n\n------------------------------------------\n\n\n")

    def play(self):
        if len(self.replaybuffer)<self.batchlen :
            return
        # print(list(self.replaybuffer))
        batch = (random.sample((self.replaybuffer), self.batchlen))
        # print(batch[0],"\n\n")
        # print(batch[1],"\n\n")
        # print(batch,"\n\n")
        states = np.array([state for state, action, state_, reward, done in batch ])
        actions = np.array([action for state, action, state_, reward, done in batch ]).astype(int)
        states_ = np.array([state_ for state, action, state_, reward, done in batch ])
        rewards = np.array([reward for state, action, state_, reward, done in batch ])
        dones = np.array([done for state, action, state_, reward, done in batch ])

        targetqs = self.tqnet.predict(states_)
        max_targetqs = np.amax(targetqs, axis =1 )
        
        act_targetqs = rewards + self.gamma*max_targetqs*(1-dones)

        currentqs = self.qnet.predict(states)
        currentqs[np.arange(self.batchlen), actions] = act_targetqs
        print("heree")

        with tf.GradientTape() as tape:
            # print(states)
            qs = self.qnet(np.array(states))
            loss = tf.reduce_mean(tf.square(currentqs - qs))
        
        gradients=tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.qnet.trainable_variables))

        self.updates+=1
        if self.updates%self.updatefreq ==0:
            self.tqnet.set_weights(self.qnet.get_weights())
            self.qnet.save("qnet.h5")
            self.tqnet.save("tarqnet.h5")

    def epsilon_decay(self):
        self.epsilon=max(self.minepsilon, self.epsilon*self.decayrate)
        
