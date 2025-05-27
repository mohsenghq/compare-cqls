import d3rlpy
import torch
import numpy as np
from d3rlpy.algos import CQLConfig
from d3rlpy.optimizers import AdamFactory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='halfcheetah_dataset.npz')
parser.add_argument('--n_steps', type=int, default=1_000_000)
args = parser.parse_args()

class MISACQL(d3rlpy.algos.CQL):
    def __init__(self, config, action_size, device='cpu', enable_ddp=False, *args, **kwargs):
        super().__init__(config, device, enable_ddp, *args, **kwargs)
        # Define a standard Gaussian prior for actions
        self.prior = torch.distributions.MultivariateNormal(
                      torch.zeros(action_size, device=self._device),
                      torch.eye(action_size, device=self._device)
                  )
        # Hyperparameters for mutual information regularization
        self.alpha1 = 1.0  # Weight for critic MI term
        self.alpha2 = 1.0  # Weight for actor MI term

    def _compute_critic_loss(self, batch):
        # Compute standard CQL critic loss
        cql_critic_loss,_ = super()._compute_critic_loss(batch)
        
        # Compute mutual information lower bound (Barber-Agakov)
        observations = batch.observations
        with torch.no_grad():
            actions = self._policy(observations).sample()
        log_pi = self._policy.log_prob(observations, actions)
        log_p = self.prior.log_prob(actions)
        mi_lower = (log_pi - log_p).mean()
        
        # Total critic loss with MI regularization
        total_loss = cql_critic_loss - self.alpha1 * mi_lower
        # print(total_loss)
        return total_loss

    def _compute_actor_loss(self, batch):
        # Sample actions from the policy
        observations = batch.observations
        actions = self._policy(observations).sample()
        
        # Compute Q-values
        q1, q2 = self._q_func(observations, actions)
        q = torch.min(q1, q2)
        # Compute mutual information term
        log_pi = self._policy.log_prob(observations, actions)
        log_p = self.prior.log_prob(actions)
        mi_term = log_pi - log_p
        # print(f'actor loss: {q_values}')
        # Actor loss with MI regularization
        actor_loss = -(q + self.alpha2 * mi_term).mean()
        
        return actor_loss

# Example usage
if __name__ == "__main__":
    # Load preprocessed dataset
    data = np.load(args.dataset_name)
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    terminals = data['terminals']

    # Create D3RLPY dataset
    d3rlpy_dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)
    config = d3rlpy.algos.CQLConfig(compile_graph=True)
    cql = MISACQL(config, device='cuda:0', enable_ddp=False, action_size=actions.shape[1])

    # Build the model
    cql.build_with_dataset(d3rlpy_dataset)

    # Train the agent
    cql.fit(d3rlpy_dataset, n_steps=args.n_steps)

    # Save the model
    cql.save_model('misa_cql_halfcheetah.d3')