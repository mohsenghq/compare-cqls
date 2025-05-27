import argparse

def get_shared_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='weights', help='Path to save/load the model')
    parser.add_argument('--dataset_name', type=str, default='halfcheetah_dataset.npz', help='Dataset file name')
    parser.add_argument('--n_steps', type=int, default=1_000_000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v5', help='Gym environment name')
    return parser 