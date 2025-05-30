import minari
import numpy as np
import d3rlpy
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='mujoco/halfcheetah/medium-v0')
args = parser.parse_args()

# modify dataset name for different os
args.dataset_name = args.dataset_name.replace('/', '\\') if os.name == 'nt' else args.dataset_name.replace('\\', '/')
# Load HalfCheetah dataset from Minari
dataset_name = args.dataset_name
try:
    dataset = minari.load_dataset(dataset_name)
except:
    dataset = minari.load_dataset(dataset_name, download=True)
# Convert to D3RLPY-compatible format
observations = []
actions = []
rewards = []
terminals = []

for episode in dataset:
    # Each episode is an EpisodeData object
    # Append observations, excluding the final observation
    observations.extend(episode.observations[:-1])
    actions.extend(episode.actions)
    rewards.extend(episode.rewards)
    # Combine terminations and truncations to mark episode ends
    # Terminals are True if either termination or truncation is True at the last step
    episode_terminals = [
        term or trunc for term, trunc in zip(episode.terminations, episode.truncations)
    ]
    terminals.extend([False] * (len(episode.rewards) - 1) + [episode_terminals[-1]])

# Convert lists to numpy arrays
observations = np.array(observations)
actions = np.array(actions)
rewards = np.array(rewards)
terminals = np.array(terminals)

# Create D3RLPY dataset
d3rlpy_dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

# Save dataset for reuse
np.savez('halfcheetah_dataset.npz', observations=observations, actions=actions, rewards=rewards, terminals=terminals)

# Print shapes and termination stats for verification
print(f"Observations shape: {observations.shape}")
print(f"Actions shape: {actions.shape}")
print(f"Rewards shape: {rewards.shape}")
print(f"Terminals shape: {terminals.shape}")
print(f"Number of terminations: {np.sum(terminals)}")