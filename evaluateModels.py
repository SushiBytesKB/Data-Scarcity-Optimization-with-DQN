import os
import json
import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SEED = 30
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from agent import DQNAgent
from preprocessor import preprocess_state

TEST_DATA = 'Models/robust_real_test.csv'
BASELINE_MODEL_PATH = 'Models/baseline_dqn.h5'
ADAPTIVE_MODEL_PATH = 'Models/adaptive_dqn.h5'
PRIME_TIME_HOURS = [12, 13, 19, 20]

def decode_action(action_idx):
    action_map = {
        0: {"actionType": "reject", "assignedTableCapacity": 0},
        1: {"actionType": "accept", "assignedTableCapacity": 2},
        2: {"actionType": "accept", "assignedTableCapacity": 4},
        3: {"actionType": "accept", "assignedTableCapacity": 6},
        4: {"actionType": "accept", "assignedTableCapacity": 8}
    }
    return action_map.get(action_idx, {"actionType": "reject", "assignedTableCapacity": 0})

def calculate_actual_reward(row, chosen_action):
    action_type = chosen_action['actionType']
    assigned_capacity = chosen_action['assignedTableCapacity']
    num_of_guests = row['numOfGuests']
    time_of_day = row['timeOfDay']
    occupancy = row['occupancy']
    customer_avg_spend = row['customerAvgSpend']

    if action_type == 'accept' and assigned_capacity < num_of_guests:
        return -100  

    wasted_seats = max(0, assigned_capacity - num_of_guests)

    if action_type == 'reject':
        price_paid = 0
    else:
        price_paid = round(customer_avg_spend * num_of_guests, 2)

    opportunity_cost = 0
    if action_type == 'accept':
        opportunity_cost += (wasted_seats * 15)
        if time_of_day in PRIME_TIME_HOURS and occupancy > 0.6:
            opportunity_cost += 50
            if occupancy > 0.9:
                opportunity_cost += 50

    return price_paid - opportunity_cost

def plot_loss_curves():
    print("\nGenerating training loss curves graph")
    
    if not os.path.exists('baseline_history.json') or not os.path.exists('adaptive_history.json'):
        print("History files not found. Run train_models.py first.")
        return

    with open('baseline_history.json', 'r') as f:
        base_hist = json.load(f)
    with open('adaptive_history.json', 'r') as f:
        adapt_hist = json.load(f)

    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot Baseline Loss
    plt.plot(base_hist['val_loss'], label='Baseline Model (Validation Loss)', color='#ff7f0e', linestyle='--')
    
    # Plot Adaptive Loss
    plt.plot(adapt_hist['val_loss'], label='Adaptive Model (Validation Loss)', color='#1f77b4', linewidth=2.5)

    plt.title('Validation Loss During Training\n(Impact of Early Stopping)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_loss_curves.png')
    print("Success! Training loss graph saved as 'training_loss_curves.png'")

def main():
    print("Starting Comparative Analysis")

    if not os.path.exists(TEST_DATA):
        print(f"Error: Could not find test dataset {TEST_DATA}")
        return

    print("Loading models...")
    baseline_agent = DQNAgent()
    baseline_agent.load(BASELINE_MODEL_PATH)

    adaptive_agent = DQNAgent()
    adaptive_agent.load(ADAPTIVE_MODEL_PATH)

    df = pd.read_csv(TEST_DATA)
    print(f"Evaluating models on {len(df)} unseen ground-truth logs\n")

    baseline_cumulative_reward = 0
    adaptive_cumulative_reward = 0
    baseline_history = []
    adaptive_history = []

    for index, row in df.iterrows():
        state_vector = preprocess_state(row)

        baseline_q_values = baseline_agent.model.predict(state_vector, verbose=0)[0]
        baseline_action = decode_action(np.argmax(baseline_q_values))
        baseline_cumulative_reward += calculate_actual_reward(row, baseline_action)
        baseline_history.append(baseline_cumulative_reward)

        adaptive_q_values = adaptive_agent.model.predict(state_vector, verbose=0)[0]
        adaptive_action = decode_action(np.argmax(adaptive_q_values))
        adaptive_cumulative_reward += calculate_actual_reward(row, adaptive_action)
        adaptive_history.append(adaptive_cumulative_reward)

    print("\nFinal Results")
    print(f"Baseline Model Total Reward: ${baseline_cumulative_reward:,.2f}")
    print(f"Adaptive Model Total Reward: ${adaptive_cumulative_reward:,.2f}")

    # Generate Original Reward Graph
    print("\nGenerating comparative reward graph...")
    plt.figure(figsize=(10, 6), dpi=300) 
    plt.plot(baseline_history, label='Baseline Model (Sparse Data)', color='#ff7f0e', linewidth=2, linestyle='--')
    plt.plot(adaptive_history, label='Adaptive Model (Hybrid Data)', color='#1f77b4', linewidth=2.5)
    plt.title('Cumulative Reward on Unseen Data\n(Overcoming the Cold Start Problem)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Number of Evaluated Customer Interactions', fontsize=12)
    plt.ylabel('Cumulative Reward ($)', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.fill_between(range(len(df)), baseline_history, adaptive_history, color='green', alpha=0.1, label='Performance Gap')
    plt.tight_layout()
    plt.savefig('comparative_analysis.png')
    print("Success! Reward graph saved as 'comparative_analysis.png'")

    # Generate the new Loss Curve Graph
    plot_loss_curves()

if __name__ == "__main__":
    main()