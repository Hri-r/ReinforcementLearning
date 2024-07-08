import tensorflow as tf
# import keras
from tensorflow import keras
import numpy as np
from networks import ActorNet, CriticNet

class PPOagent():
    def __init__(self, statd,actd, gamma = 0.99, alpha_a = 0.00001, alpha_c = 0.0001, lam = 0.95, clipping_ratio = 0.2, updates = 20, reg = 0.01, minepsilon = 0.0001, decayrate = 0.995, epsilon = 0.1):
        self.statd = statd
        self.actd =actd
        self.gamma = gamma
        self.alpha_a = alpha_a
        self.alpha_c = alpha_c
        self.lam = lam
        self.clip = clipping_ratio
        self.reg = reg

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.alpha_a)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.alpha_c)

        self.actor = ActorNet(self.statd, self.actd)
        self.critic = CriticNet(self.statd)
        self.num_updates = updates

        self.minepsilon = minepsilon
        self.epsilon = epsilon
        self.decay_rate = decayrate

    def selectAction(self, state):
        # state = np.reshape(state, (1, -1))
        if(np.random.random()>self.epsilon):
            state = np.expand_dims(state, axis = 0)
            # state = np.expand_dims(state, axis = -1)
            probs = self.actor(state).numpy()[0]
            action = np.random.choice(self.actd, p = probs)
        else:
            action = np.random.choice(self.actd)
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

        # print(advantages)

        advantages = (advantages - np.mean(advantages))/(np.std(advantages) - 1e-10)

        old_probs = self.actor(states)

        old_probs = np.array([old_probs[i, actions[i]] for i in range(len(actions))])

        old_critic_loss = tf.Variable(0.0)

        for _ in range(self.num_updates):
            with tf.GradientTape() as tape:
                probs = self.actor(states, training = True)
                action_probs = tf.gather_nd(probs, np.array([[i, actions[i]] for i in range(len(actions))]))
                action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)
                old_probs = tf.clip_by_value(old_probs, 1e-8, 1.0)

                r = action_probs/old_probs
                clipped_r = tf.clip_by_value(r, 1-self.clip, 1+self.clip)
                # print(clipped_r)
                # print(advantages)
                entropy_loss = -self.reg *tf.reduce_mean(-tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1))
                actor_loss = -tf.reduce_mean(tf.minimum(r*advantages, clipped_r*advantages)) + entropy_loss
            
            tf.print("action_probs:", action_probs)
            tf.print("ratios:", r)
            tf.print("clipped_ratios:", clipped_r)
            tf.print("entropy_loss:", entropy_loss)
            tf.print("actor_loss:", actor_loss)
            
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads = [tf.clip_by_norm(g, 0.5) for g in actor_grads]
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            with tf.GradientTape() as tape:
                critic_loss = tf.reduce_mean(tf.square(returns - self.critic(states, training = True)))
            
            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            critic_grads = [tf.clip_by_norm(g, 0.5) for g in critic_grads]
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

            if(tf.math.abs(critic_loss-old_critic_loss)<10-8):
                break

            old_critic_loss = actor_loss

    def save_models(self, actor_path='actor.keras', critic_path='critic.keras'):
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_models(self, actor_path='actor.keras', critic_path='critic.keras'):
        self.actor = keras.models.load_model(actor_path, custom_objects={'ActorNet': ActorNet})
        self.critic = keras.models.load_model(critic_path, custom_objects={'CriticNet': CriticNet})

    def epsilon_decay(self):
        self.epsilon=max(self.minepsilon, self.epsilon*self.decay_rate)
