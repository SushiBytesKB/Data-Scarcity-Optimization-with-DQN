import os
import json
import random
import numpy as np
import tensorflow as tf

SEED = 30
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from agent import DQNAgent

BASELINE_DATA = 'Models/sparse_real_train.csv'
HYBRID_DATA = 'Models/hybrid_training_dataset.csv'

BASELINE_MODEL_NAME = 'Models/baseline_dqn.h5'
ADAPTIVE_MODEL_NAME = 'Models/adaptive_dqn.h5'

def main():
    print("Starting Offline RL Training Pipeline\n")

    if not os.path.exists(BASELINE_DATA) or not os.path.exists(HYBRID_DATA):
        print("Error: Training data not found.")
        return

    # Train Baseline Model
    print("Training Baseline Model (Method 1)")
    baseline_agent = DQNAgent()
    # Assume 100 rows of data - 100/16 = 6.25 learning opportunities per epoch
    # 6.25 * 100 = 625
    baseline_history = baseline_agent.train_from_csv(BASELINE_DATA, epochs=100, batch_size=16)
    baseline_agent.save(BASELINE_MODEL_NAME)
    
    # Save training history
    with open('baseline_history.json', 'w') as f:
        json.dump(baseline_history, f)
    print("------------------------------------------\n")

    # Train Adaptive Model
    print("Training Adaptive Model (Method 2)")
    adaptive_agent = DQNAgent()
    # Assume 900 rows of data - 900/128 = 7.03125 learning opportunities per epoch
    # 7.03125 * 100 = 703.125
    adaptive_history = adaptive_agent.train_from_csv(HYBRID_DATA, epochs=100, batch_size=128)
    adaptive_agent.save(ADAPTIVE_MODEL_NAME)
    
    # Save training history
    with open('adaptive_history.json', 'w') as f:
        json.dump(adaptive_history, f)
    print("----------------------------------------------------\n")

    print("Pipeline Complete!")

if __name__ == "__main__":
    main()