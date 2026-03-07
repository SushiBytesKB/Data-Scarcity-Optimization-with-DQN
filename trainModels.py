import os
from agent import DQNAgent

BASELINE_DATA = 'Models/sparse_real_train.csv'
HYBRID_DATA = 'Models/hybrid_training_dataset.csv'

BASELINE_MODEL_NAME = 'Models/baseline_dqn.h5'
ADAPTIVE_MODEL_NAME = 'Models/adaptive_dqn.h5'

def main():
    print("\tStarting Offline RL Training Pipeline\n")

    if not os.path.exists(BASELINE_DATA) or not os.path.exists(HYBRID_DATA):
        print("Error: Please ensure you have generated the CSVs using both components")
        return

    # Train the Baseline Model (Method 1)
    print("Training Baseline Model (Method 1)")
    print(f"Dataset: {BASELINE_DATA} (100 Sparse Logs)")
    baseline_agent = DQNAgent()
    
    baseline_agent.train_from_csv(BASELINE_DATA, epochs=100, batch_size=16)
    baseline_agent.save(BASELINE_MODEL_NAME)
    print("------------------------------------------\n")

    # Train the Adaptive Synthetic Model (Method 2)
    print("Training Adaptive Model (Method 2)")
    print(f"Dataset: {HYBRID_DATA} (1100 Hybrid Logs)")
    adaptive_agent = DQNAgent()
    
    adaptive_agent.train_from_csv(HYBRID_DATA, epochs=50, batch_size=32)
    adaptive_agent.save(ADAPTIVE_MODEL_NAME)
    print("----------------------------------------------------\n")

    print("Pipeline Complete!")
    print(f"Models successfully saved as '{BASELINE_MODEL_NAME}' and '{ADAPTIVE_MODEL_NAME}'.")

if __name__ == "__main__":
    main()