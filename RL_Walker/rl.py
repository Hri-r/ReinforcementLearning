from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from LegEnv import LegEnv

# Wrap with a lambda that includes render_mode
env = make_vec_env(lambda: LegEnv(render_mode="human"), n_envs=1)

from stable_baselines3.common.callbacks import BaseCallback

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=10, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.training_env.render()
        return True

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, callback=RenderCallback())
