import os
import time
from stable_baselines3 import SAC
from Env3d import Env3d  # Make sure your environment file is accessible

# --- Configuration ---

# The directory where your models are saved
SAVE_DIR = "logs/best_model"
# The full path to the specific model file you want to test.
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.zip") 
# MODEL_PATH = "load_sac/firststepagain.zip"

# --- Main Inference Logic ---

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# 1. Create the Environment
env = Env3d(render_mode="human")

# 2. Load the Trained Model
print(f"Loading model from: {MODEL_PATH}")

# --- THE FIX ---
# When loading a model that was saved with a fixed entropy coefficient,
# you must tell the .load() function to create a model that also expects
# a fixed entropy coefficient. We do this by passing the ent_coef argument.
# The exact float value doesn't matter, just that it's not 'auto'.
try:
    model = SAC.load(MODEL_PATH, env=env)
except:
    model = SAC.load(MODEL_PATH, env=env, ent_coef=0.01) 
# --- END OF FIX ---

print("Model loaded successfully.")

# 3. The Inference Loop
print("Running inference... Press Ctrl+C to exit.")

obs, info = env.reset()
target_frame_duration = 0.5/ env.metadata['render_fps']

try:
    while True:
        loop_start_time = time.time()

        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()

        elapsed_time = time.time() - loop_start_time
        sleep_duration = target_frame_duration - elapsed_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)

except KeyboardInterrupt:
    print("\nInference stopped by user.")
finally:
    env.close()
