import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from Env3d import Env3d

# --- Configuration ---
TOTAL_TIMESTEPS = 1000000
SAVE_FREQ = 50000
SAVE_DIR = "ppo_models"

# **IMPORTANT**: If you choose to load, this is the model that will be loaded.
# Make sure to update this path to your desired model file before running.
MODEL_TO_LOAD = os.path.join("load_ppo", "load.zip")

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)

# Your SaveCallback class (unchanged)
class SaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"ppo_model_{self.n_calls}_steps")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model at {self.n_calls} steps to {model_path}")
        return True

# --- Main Logic ---

# 1. Create the Environment
env = make_vec_env(lambda: Env3d(render_mode="human"), n_envs=1)

# 2. Ask the user if they want to load a model
should_load = input("Do you want to load a saved model? (y/n): ").lower().strip()

# 3. Create or Load the model based on user input
if should_load.startswith('y'):
    print(f"--- Attempting to load model from: {MODEL_TO_LOAD} ---")
    if os.path.exists(MODEL_TO_LOAD):
        model = PPO.load(MODEL_TO_LOAD, env=env)
        # We set reset_num_timesteps=False to continue the step count
        reset_timesteps = False
        print("--- Model loaded. Continuing training. ---")
    else:
        print(f"--- ERROR: Model file not found. Starting new training run. ---")
        model = PPO("MlpPolicy", env, verbose=1)
        reset_timesteps = True
else:
    print("--- Starting a new training run. ---")
    model = PPO("MlpPolicy", env, verbose=1)
    reset_timesteps = True


# 4. Train the model
save_callback = SaveCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=save_callback,
    reset_num_timesteps=reset_timesteps # This is either True or False based on the logic above
)

# Save the final version after training is complete
model.save(os.path.join(SAVE_DIR, "ppo_model_final.zip"))
print("--- Training complete. Final model saved. ---")