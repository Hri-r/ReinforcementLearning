import keras
import tensorflow as tf
import numpy as np

class ActorNet(keras.Model):
    def __init__(self, stat_dims, action_dim, hidden_units = (64,64), name = 'actor_network',*args,  **kwargs):
        super(ActorNet, self).__init__(name = name,*args, **kwargs)

        self.stat_dim = stat_dims
        self.act_dim = action_dim
        self.hidden_units = hidden_units

        self.actor_network = self.build_act_net()
    
    
    def build_act_net(self):
        model = keras.Sequential()

        model.add(keras.layers.InputLayer(shape= self.stat_dim))

        for units in self.hidden_units:
            model.add(keras.layers.Dense(units, activation='relu'))

        model.add(keras.layers.Dense(self.act_dim, activation="softmax"))

        return model
    
    def call(self, state, training = False):
        return self.actor_network(state, training = training)
    
    def get_config(self):
        config = super(ActorNet, self).get_config()
        config.update({
            'stat_dims': self.stat_dim,
            'action_dim': self.act_dim,
            'hidden_units': self.hidden_units,
        })
        return config

class CriticNet(keras.Model):
    def __init__(self, stat_dims, hidden_units = (64,64, 64), name = 'critic_network', *args, **kwargs):
        super(CriticNet, self).__init__(name = name,*args, **kwargs)

        self.stat_dim = stat_dims
        self.hidden_units = hidden_units

        self.critic_network = self.build_crit_net()
    
    def build_crit_net(self):
        model = keras.Sequential()
        model.add(keras.layers.InputLayer(shape= self.stat_dim))

        for units in self.hidden_units:
            model.add(keras.layers.Dense(units, activation='relu'))

        model.add(keras.layers.Dense(1, activation=None))

        return model
    
    def call(self, state, training = False):
        return self.critic_network(state, training = training)

    def get_config(self):
        config = super(CriticNet, self).get_config()
        config.update({
            'stat_dims': self.stat_dim,
            'hidden_units': self.hidden_units,
        })
        return config    