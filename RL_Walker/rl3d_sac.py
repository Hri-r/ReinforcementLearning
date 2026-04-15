import os
import time
import torch as th
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.logger import configure
from Env3d import Env3d

# --- Configuration ---
TOTAL_TIMESTEPS = 2000000
SAVE_FREQ = 10000
EVAL_FREQ = 5000
SAVE_DIR = "sac_models"
LOG_DIR = "logs"
RUN_NAME = "SAC_WalkingBot_Run1" 

MODEL_TO_LOAD = os.path.join("load_sac", "firststepagain.zip")

# --- Setup ---
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(f"{LOG_DIR}/tensorboard", exist_ok=True)

# --- Custom Callbacks ---

class SaveCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"sac_model_{self.model.num_timesteps}_steps")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model at {self.model.num_timesteps} steps to {self.model.num_timesteps}")
        return True

class EntCoefSchedulerCallback(BaseCallback):
    """
    A callback to schedule the entropy coefficient (ent_coef).
    Starts with a fixed value and switches to 'auto' at a specified step.
    """
    def __init__(self, switch_step: int, fixed_ent_coef: float, verbose: int = 1):
        super().__init__(verbose)
        self.switch_step = switch_step
        self.fixed_ent_coef = fixed_ent_coef
        self.switched_to_auto = False

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        It overrides the initial ent_coef setting.
        """
        if self.verbose > 0:
            print(f"Callback: Setting initial ent_coef to fixed value: {self.fixed_ent_coef}")
        # Disable automatic tuning and set a fixed value
        self.model.ent_coef_optimizer = None
        self.model.log_ent_coef = th.log(th.ones(1, device=self.model.device) * self.fixed_ent_coef).requires_grad_(False)
        
        # --- FIX FOR OLDER SB3 VERSIONS ---
        # Some versions expect the `ent_coef_tensor` attribute to be explicitly set
        # when not in 'auto' mode, instead of relying on the property.
        self.model.ent_coef_tensor = th.exp(self.model.log_ent_coef)
        # --- END OF FIX ---


    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        """
        if not self.switched_to_auto and self.model.num_timesteps >= self.switch_step:
            if self.verbose > 0:
                print(f"Callback: Switching ent_coef to 'auto' at step {self.model.num_timesteps}")
            
            if self.model.target_entropy == 'auto':
                 self.model.target_entropy = -np.prod(self.model.action_space.shape).astype(np.float32)

            self.model.log_ent_coef = th.log(th.ones(1, device=self.model.device) * self.fixed_ent_coef).requires_grad_(True)
            
            actor_lr = self.model.actor.optimizer.param_groups[0]['lr']
            self.model.ent_coef_optimizer = th.optim.Adam([self.model.log_ent_coef], lr=actor_lr)

            # --- FIX FOR OLDER SB3 VERSIONS ---
            # Remove the explicitly set attribute so the model reverts to using the property
            if hasattr(self.model, "ent_coef_tensor"):
                del self.model.ent_coef_tensor
            # --- END OF FIX ---

            self.switched_to_auto = True
        return True


# --- Main Logic ---

# 1. Create the Environments
seed = int(time.time())
print(f"USING SEED: {seed}")
env = make_vec_env(lambda: Env3d(render_mode=None), n_envs=1, seed=seed)
eval_env = make_vec_env(lambda: Env3d(render_mode=None), n_envs=1, seed=seed + 1)

# 2. Ask the user if they want to load a model
should_load = input("Do you want to load a saved model? (y/n): ").lower().strip()

# 3. Create or Load the model based on user input
if should_load.startswith('y'):
    print(f"--- Attempting to load model from: {MODEL_TO_LOAD} ---")
    if os.path.exists(MODEL_TO_LOAD):
        # When loading, the model should have been created with ent_coef='auto'
        # for the callback to work correctly when switching back.
        model = SAC.load(MODEL_TO_LOAD, env=env)
        print("--- Model loaded successfully. ---")

        print("Re-attaching logger to the loaded model...")
        new_logger = configure(f"{LOG_DIR}/tensorboard/", ["tensorboard"])
        model.set_logger(new_logger)

        last_step_str = input(f"Model's internal clock is at {model.num_timesteps}. Enter the last timestep from the previous run to continue the graph (e.g., 1105000): ")
        try:
            last_step = int(last_step_str)
            model.num_timesteps = last_step
            print(f"Manually set timestep counter to {model.num_timesteps}.")
        except ValueError:
            print("Invalid number. Timestep counter not updated.")

        lr_input = input("Enter new learning rate (or press Enter to keep existing): ")
        try:
            new_lr = float(lr_input)
            print(f"Setting new learning rate for optimizers to: {new_lr}")
            model.actor.optimizer.param_groups[0]['lr'] = new_lr
            model.critic.optimizer.param_groups[0]['lr'] = new_lr
            if model.ent_coef_optimizer is not None:
                model.ent_coef_optimizer.param_groups[0]['lr'] = new_lr
        except ValueError:
            print(f"--- Invalid input. Model loaded with its existing LR. ---")
        
        reset_timesteps = False
    else:
        print(f"--- ERROR: Model file not found. Starting new training run. ---")
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"{LOG_DIR}/tensorboard/", ent_coef='auto')
        reset_timesteps = True
else:
    print("--- Starting a new training run. ---")
    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"{LOG_DIR}/tensorboard/", ent_coef='auto')
    reset_timesteps = True

# 4. Set up Callbacks
save_callback = SaveCallback(save_freq=SAVE_FREQ, save_path=SAVE_DIR)
eval_callback = EvalCallback(eval_env,
                           best_model_save_path=os.path.join(LOG_DIR, 'best_model'),
                           log_path=os.path.join(LOG_DIR, 'results'),
                           eval_freq=EVAL_FREQ,
                           deterministic=True,
                           render=False)

ent_coef_callback = EntCoefSchedulerCallback(switch_step=model.num_timesteps + 50000, fixed_ent_coef=0.01)

callback_list = CallbackList([save_callback, eval_callback, ent_coef_callback])

# 5. Train the model
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callback_list,
    reset_num_timesteps=reset_timesteps,
    tb_log_name=RUN_NAME
)

# Save the final version after training is complete
model.save(os.path.join(SAVE_DIR, "sac_model_final.zip"))
print("--- Training complete. Final model saved. ---")
