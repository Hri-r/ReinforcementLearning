import numpy as np
import matplotlib.pyplot as plt

# Path to your evaluations file
log_file = "logs/results/evaluations.npz"

# Load the data
data = np.load(log_file)

timesteps = data['timesteps']
results = data['results'] # This is a 2D array: (num_evaluations, num_episodes)

# Calculate the mean and standard deviation of rewards for each evaluation
mean_reward = np.mean(results, axis=1)
std_reward = np.std(results, axis=1)

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(timesteps, mean_reward, label="Mean Reward")

# Create the shaded area for the standard deviation
plt.fill_between(timesteps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2, color='blue')

plt.title("Agent Performance Over Time")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.grid(True)
plt.legend()
plt.show()