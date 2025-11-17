# Offline Reinforcement Learning with CQL and MISA

This project implements and evaluates several offline reinforcement learning algorithms, including Conservative Q-Learning (CQL) and its variants. The primary goal is to train agents on the HalfCheetah-v5 environment using the d3rlpy library.

## Training Scripts

* **`train_cql.py`**: This script trains a standard CQL agent.
* **`train_misa_cql.py`**: This script trains a CQL agent with MISA (Mixture of Iso-Gaussian Policies).
* **`train_misa_cql_heydari.py`**: This script trains a CQL agent with a MISA implementation based on Heydari's work.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Install dependencies:**
   ```bash
   pip install d3rlpy torch numpy minari gym
   ```

3. **Download and preprocess the dataset:**
   ```bash
   python load_dataset.py
   ```

## Usage

### Training

To train an agent, run one of the training scripts:

```bash
# Train a standard CQL agent
python train_cql.py

# Train a CQL agent with MISA
python train_misa_cql.py

# Train a CQL agent with MISA (Heydari's implementation)
python train_misa_cql_heydari.py
```

### Evaluation

To evaluate a trained agent, run the `evaluate_agents.py` script:

```bash
python evaluate_agents.py --model_path weights/cql_halfcheetah.d3
```

Run this code in colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Sc2l4KdxFT10AJs79108xV-7AfOCJ3QE?usp=sharing)
