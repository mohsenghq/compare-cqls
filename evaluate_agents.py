import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import d3rlpy
from d3rlpy.algos import CQL
import torch

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration 
MODEL_PATHS = [
    'cql_halfcheetah.d3',  # Update with your actual file paths
    'misa_cql_halfcheetah.d3',
    'misa_cql_halfcheetah_heydari.d3'
]
MODEL_NAMES = ['CQL', 'MISA', 'MISA heydari']  # Update with descriptive names
NUM_EPISODES = 100  # Number of evaluation episodes per model
ENV_NAME = 'HalfCheetah-v4'

def load_model(path):
    """Load a CQL model from the given path."""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cql = d3rlpy.algos.CQLConfig(compile_graph=False).create(device=device)

    # Build the model
    cql.build_with_env(gym.make(ENV_NAME))
    cql.load_model(path)
    return cql

def evaluate_model(model, env_name, n_episodes=10):
    """Evaluate a model on the given environment for n_episodes."""
    env = gym.make(env_name, render_mode=None)  # No rendering during evaluation
    
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        observation, _ = env.reset(seed=RANDOM_SEED)
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # print(observation)
            action = model.predict(np.array([observation]))[0]
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    return {
        'rewards': episode_rewards,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'lengths': episode_lengths,
        'mean_length': np.mean(episode_lengths)
    }

def render_model(model, env_name, n_steps=1000):
    """Render a model's performance on the environment."""
    try:
        # Try to create environment with rendering
        env = gym.make(env_name, render_mode='human')
        
        observation, _ = env.reset(seed=RANDOM_SEED)
        episode_reward = 0
        
        for _ in range(n_steps):
            action = model.predict([observation])[0]
            observation, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                print(f"Episode finished with reward: {episode_reward}")
                break
        
        env.close()
    except Exception as e:
        print(f"Rendering failed: {e}. Rendering may not be available in your environment.")

def visualize_comparison(results, model_names):
    """Create visualizations comparing model performance."""
    plt.figure(figsize=(12, 8))
    
    # Plot reward distribution
    plt.subplot(2, 1, 1)
    plt.boxplot([r['rewards'] for r in results], labels=model_names)
    plt.title('Reward Distribution Across Episodes')
    plt.ylabel('Total Episode Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot mean rewards with error bars
    plt.subplot(2, 1, 2)
    means = [r['mean_reward'] for r in results]
    stds = [r['std_reward'] for r in results]
    x = np.arange(len(model_names))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10)
    plt.xticks(x, model_names)
    plt.title('Mean Episode Reward')
    plt.ylabel('Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    # Load models
    print("Loading models...")
    models = [load_model(path) for path in MODEL_PATHS]
    
    # Evaluate models
    results = []
    for i, model in enumerate(models):
        print(f"Evaluating {MODEL_NAMES[i]}...")
        result = evaluate_model(model, ENV_NAME, NUM_EPISODES)
        results.append(result)
        print(f"  Mean reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Mean episode length: {result['mean_length']:.2f}")
    
    # Compare results
    print("\nModel Comparison:")
    for i, result in enumerate(results):
        print(f"{MODEL_NAMES[i]}: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    
    # Visualize comparison
    visualize_comparison(results, MODEL_NAMES)
    
    # Optional: Render one episode for each model
    # render_option = input("Would you like to render a demonstration of each model? (y/n): ")
    # if render_option.lower() == 'y':
    #     for i, model in enumerate(models):
    #         print(f"\nRendering {MODEL_NAMES[i]}...")
    #         render_model(model, ENV_NAME)

if __name__ == "__main__":
    main()