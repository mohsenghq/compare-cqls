import d3rlpy
import numpy as np
from d3rlpy.optimizers import AdamFactory
import torch

# Load preprocessed dataset
data = np.load('halfcheetah_dataset.npz')
observations = data['observations']
actions = data['actions']
rewards = data['rewards']
terminals = data['terminals']

# Create D3RLPY dataset
d3rlpy_dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cql = d3rlpy.algos.CQLConfig(compile_graph=True).create(device=device)

# Build the model
cql.build_with_dataset(d3rlpy_dataset)

# Train CQL
cql.fit(d3rlpy_dataset, n_steps=100000)

# Save model
cql.save_model('cql_halfcheetah.d3')