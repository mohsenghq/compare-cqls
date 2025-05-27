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
    def __init__(self, config, device='cpu', enable_ddp=False, *args, **kwargs):
        super().__init__(config, device, enable_ddp, *args, **kwargs)
        # Hyperparameters for mutual information regularization
        self.alpha1 = 1.0  # Weight for critic MI term
        self.alpha2 = 1.0  # Weight for actor MI term

    def _sample_p_psi(self, observations, num_samples=5, step_size=0.1, num_burnin=10):
        """Sample actions from p_psi(a|s) using Metropolis-Hastings."""
        def log_prob_func(s, a):
            a = a.requires_grad_(False)  # No gradients through sampling
            log_pi = self._policy.log_prob(s.unsqueeze(0), a.unsqueeze(0)).squeeze()
            q_value = self._q_func(s.unsqueeze(0), a.unsqueeze(0))[0].squeeze()  # Use first Q-function
            return log_pi + q_value

        samples_batch = []
        for s in observations:
            a0 = self._policy(s.unsqueeze(0)).sample().squeeze()
            samples = []
            a = a0.clone()
            log_p_current = log_prob_func(s, a)
            total_steps = num_samples + num_burnin
            for _ in range(total_steps):
                a_proposed = a + step_size * torch.randn_like(a)
                log_p_proposed = log_prob_func(s, a_proposed)
                accept_prob = torch.exp(log_p_proposed - log_p_current).clamp(max=1.0)
                if torch.rand(1, device=self._device) < accept_prob:
                    a = a_proposed
                    log_p_current = log_p_proposed
                if _ >= num_burnin:
                    samples.append(a.clone().detach())
            samples_batch.append(torch.stack(samples))
        return torch.stack(samples_batch)  # Shape: [batch_size, num_samples, action_size]

    def _compute_critic_loss(self, batch):
        # Compute standard CQL critic loss
        cql_critic_loss, _ = super()._compute_critic_loss(batch)
        
        # Compute new MI term based on gradient formula
        observations = batch.observations
        actions_d = batch.actions  # Dataset actions
        q_d = self._q_func(observations, actions_d)[0]  # Use first Q-function
        
        # Sample actions from p_psi(a|s)
        sampled_actions = self._sample_p_psi(observations, num_samples=5)  # [batch_size, num_samples, action_size]
        batch_size = observations.size(0)
        q_sampled = torch.stack([
            self._q_func(observations, sampled_actions[:, k])[0]
            for k in range(sampled_actions.size(1))
        ]).mean(dim=0)  # Average over samples
        
        # MI term approximation
        mi_term_critic = q_d.mean() - q_sampled.mean()
        
        # Total critic loss with MI regularization
        total_loss = cql_critic_loss - self.alpha1 * mi_term_critic
        return total_loss

    def _compute_actor_loss(self, batch):
        # Sample actions from the policy for Q-value computation
        observations = batch.observations
        actions_pi = self._policy(observations).sample()
        q1, q2 = self._q_func(observations, actions_pi)
        q = torch.min(q1, q2)
        
        # Compute MI term using dataset actions and sampled actions
        actions_d = batch.actions  # Dataset actions
        log_pi_d = self._policy.log_prob(observations, actions_d)
        
        # Sample actions from p_psi(a|s)
        sampled_actions = self._sample_p_psi(observations, num_samples=5)
        log_pi_sampled = torch.stack([
            self._policy.log_prob(observations, sampled_actions[:, k])
            for k in range(sampled_actions.size(1))
        ]).mean(dim=0)  # Average over samples
        
        # MI term approximation
        mi_term_actor = (log_pi_d - log_pi_sampled).mean()
        
        # Actor loss with MI regularization
        actor_loss = -(q.mean() + self.alpha2 * mi_term_actor)
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
    config = CQLConfig(compile_graph=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cql = MISACQL(config, device=device, enable_ddp=False)

    # Build the model
    cql.build_with_dataset(d3rlpy_dataset)

    # Train the agent
    cql.fit(d3rlpy_dataset, n_steps=args.n_steps)

    # Save the model
    cql.save_model('misa_cql_halfcheetah_heydari.d3')