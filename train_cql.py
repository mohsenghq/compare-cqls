import d3rlpy
import numpy as np
from d3rlpy.optimizers import AdamFactory
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='halfcheetah_dataset.npz')
parser.add_argument('--n_steps', type=int, default=1_000_000)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

# Load preprocessed dataset
data = np.load(args.dataset_name)
observations = data['observations']
actions = data['actions']
rewards = data['rewards']
terminals = data['terminals']

# Create D3RLPY dataset
d3rlpy_dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cql = d3rlpy.algos.CQLConfig(compile_graph=True, batch_size=args.batch_size).create(device=device)

# Build the model
cql.build_with_dataset(d3rlpy_dataset)

# Train CQL
cql.fit(d3rlpy_dataset, n_steps=args.n_steps)

# Save model
cql.save_model('weights/v2/cql_halfcheetah.d3')