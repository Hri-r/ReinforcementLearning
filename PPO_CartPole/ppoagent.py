import tensorflow as tf
# import keras
from tensorflow import keras
import numpy as np
from networks import ActorNet, CriticNet

class PPOagent():
    def __init__(self, statd,actd, gamma = 0.99, alpha = 0.0003, lam = 0.95, clipping_ratio = 0.2, updates = 100):
        self.statd = statd
        self.actd =actd
        self.gamma = gamma
        self.alpha = alpha
        self.lam = lam
        self.clip = clipping_ratio

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.alpha)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.alpha)

        self.actor = ActorNet(self.statd, self.actd)
        self.critic = CriticNet(self.statd)
        self.num_updates = updates

    def selectAction(self, state):
        state = np.reshape(state, (1, -1))
        probs = self.actor(state).numpy()[0]
        action = np.random.choice(self.actd, p = probs)
        return action
    
    def calcAdvantage(self, rewards, values, values_, dones ):
        advantages= []
        
        # deltas = rewards + self.gamma*values_*(1-np.array(dones)) - values
        
        gae = 0
        for i in reversed(range(len(rewards))):
            # print("\n\n\n")
            # print(values_)
            delta = rewards[i] + self.gamma*values_[i]*(1-dones[i]) - values[i]
            gae = delta  + self.gamma*self.lam*gae
            advantages.insert(0,gae)
        return advantages
    
    def update(self, states, states_, actions, rewards, dones):
        values = self.critic.predict(states).flatten()
        values_ = self.critic.predict(states_).flatten()
        
        advantages= self.calcAdvantage(rewards, values, values_, dones)
        # advantages = np.array(advantages)
        returns = advantages + values.flatten()

        advantages = (advantages - np.mean(advantages))/(np.std(advantages) - 1e-10)

        old_probs = self.actor(states)
        # print(old_probs.shape)
        old_probs = np.array([old_probs[i, actions[i]] for i in range(len(actions))])

        for _ in range(self.num_updates):
            with tf.GradientTape() as tape:
                probs = self.actor(states, training = True)
                probs = tf.gather_nd(probs, np.array([[i, actions[i]] for i in range(len(actions))]))

                r = probs/old_probs
                clipped_r = tf.clip_by_value(r, 1-self.clip, 1+self.clip)
                # print(clipped_r)
                # print(advantages)
                actor_loss = -tf.reduce_mean(tf.minimum(r*advantages, clipped_r*advantages))
            
            # print(actor_loss)
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states, training = True)))
            
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def save_models(self, actor_path='actor.keras', critic_path='critic.keras'):
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_models(self, actor_path='actor.keras', critic_path='critic.keras'):
        self.actor = keras.models.load_model(actor_path, custom_objects={'ActorNet': ActorNet})
        self.critic = keras.models.load_model(critic_path, custom_objects={'CriticNet': CriticNet})

            