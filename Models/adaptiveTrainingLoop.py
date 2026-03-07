import pandas as pd
import numpy as np
import random

OUTPUT_TRAIN_FILE_NAME = 'Models/sparse_real_train.csv'
OUTPUT_ADAPTIVE_TRAIN_FILE_NAME = 'Models/hybrid_training_dataset.csv'

SYNTHETIC_RATIO = 10  # Generate X logs for every 1 real log
ALPHA = 0.15          # The learning rate for the EMA - (X * 100)% new data, ((1 - X) * 100)% old memory

# The model starts with base assumptions about the restaurant
learned_spend_mean = 80.0
learned_occ_mean = 0.9                    # occupancy
learned_guest_weights = np.ones(8) / 8.0  # Assuming party sizes 1-8 are equally likely
learned_time_weights = np.ones(14) / 14.0 # Assuming all hours (9-22) are equally busy

# Constant environment rules (not customer behavior)
RESTAURANT_TABLES = [2, 2, 2, 4, 4, 4, 4, 6, 6, 8]
PRIME_TIME_HOURS = [12, 13, 19, 20]
HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def deterministic_environment(state_action):
    # Calculate exact reward and next state based on generated state and action
    time_of_day = state_action['timeOfDay']
    num_of_guests = state_action['numOfGuests']
    occupancy = state_action['occupancy']
    action_type = state_action['actionType']
    assigned_table_capacity = state_action['assignedTableCapacity']
    customer_avg_spend = state_action['customerAvgSpend']
    is_weekend = state_action['isWeekend']

    wasted_seats = max(0, assigned_table_capacity - num_of_guests)
    
    # Calculate Actual Duration (Next State logic)
    is_lunch = 11 <= time_of_day <= 14
    base_duration = 30 if is_lunch else 45
    duration = base_duration + (num_of_guests * 5)
    if is_weekend: duration += 15
    actual_duration = max(30, duration + random.randint(-10, 15))

    # Calculate Reward (Target Q-Value)
    if action_type == 'reject':
        price_paid = 0
    else:
        check_variance = random.uniform(0.8, 1.2)
        price_paid = round(customer_avg_spend * num_of_guests * check_variance, 2)

    opportunity_cost = 0
    if action_type == 'accept':
        opportunity_cost += (wasted_seats * 15)
        if time_of_day in PRIME_TIME_HOURS and occupancy > 0.6:
            opportunity_cost += 50
            if occupancy > 0.9:
                opportunity_cost += 50

    target_q_value = round(price_paid - opportunity_cost, 2)
    
    return wasted_seats, actual_duration, target_q_value

def generate_synthetic_logs(num_logs):
    # Generates X logs using the CURRENT learned parameters
    synth_logs = []
    for _ in range(num_logs):
        # Generate State from learned distributions
        num_of_guests = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=learned_guest_weights)[0]
        time_of_day = random.choices(HOURS, weights=learned_time_weights)[0]
        is_weekend = random.choice([True, False])
        customer_visit_count = random.randint(0, 10) # Simplified for generation
        
        customer_avg_spend = max(15, np.random.normal(learned_spend_mean, 10))
        occupancy = np.clip(np.random.normal(learned_occ_mean, 0.15), 0.0, 1.0)
        
        # Randomly select an Action to explore the environment
        possible_tables = [t for t in RESTAURANT_TABLES if t >= num_of_guests]
        if not possible_tables or random.random() < 0.1:
            action_type = 'reject'
            assigned_table_capacity = 0
        else:
            action_type = 'accept'
            assigned_table_capacity = random.choice(possible_tables)

        state_action = {
            'isWeekend': is_weekend,
            'timeOfDay': time_of_day,
            'occupancy': round(occupancy, 2),
            'numOfGuests': num_of_guests,
            'customerVisitCount': customer_visit_count,
            'customerAvgSpend': round(customer_avg_spend, 2),
            'actionType': action_type,
            'assignedTableCapacity': assigned_table_capacity
        }

        # Pass through the deterministic environment
        wasted_seats, actual_duration, target_q_value = deterministic_environment(state_action)
        
        state_action['wastedSeats'] = wasted_seats
        state_action['actualDuration'] = actual_duration
        state_action['targetQValue'] = target_q_value
        
        synth_logs.append(state_action)
    
    return synth_logs

def run_adaptive_loop(real_data_path, output_path):
    global learned_spend_mean, learned_occ_mean, learned_guest_weights, learned_time_weights
    
    real_df = pd.read_csv(real_data_path)
    hybrid_dataset = []

    print(f"Starting Adaptive Loop on {len(real_df)} real logs\n")
    print("Initial Naive Spend Mean: $80.00")

    for index, row in real_df.iterrows():
        # Append the real log to the final dataset
        hybrid_dataset.append(row.to_dict())

        # Update Params (The Learning Step) - Exponential Moving Average (EMA)
        learned_spend_mean = (ALPHA * row['customerAvgSpend']) + ((1 - ALPHA) * learned_spend_mean)
        learned_occ_mean = (ALPHA * row['occupancy']) + ((1 - ALPHA) * learned_occ_mean)

        # Categorical Variables (Probability Shift)
        guest_idx = int(row['numOfGuests']) - 1
        learned_guest_weights = (1 - ALPHA) * learned_guest_weights
        learned_guest_weights[guest_idx] += ALPHA

        time_idx = HOURS.index(int(row['timeOfDay']))
        learned_time_weights = (1 - ALPHA) * learned_time_weights
        learned_time_weights[time_idx] += ALPHA

        # Generate Synthetic Data
        synthetic_batch = generate_synthetic_logs(SYNTHETIC_RATIO)
        hybrid_dataset.extend(synthetic_batch)

        # Print progress every 20 logs for the visual demo
        if (index + 1) % 20 == 0:
            print(f"Processed {index + 1} real logs. Current Learned Spend Mean: ${learned_spend_mean:.2f}")

    # Save the hybrid dataset
    final_df = pd.DataFrame(hybrid_dataset)
    final_df.to_csv(output_path, index=False)
    
    print(f"\nAdaptive loop complete! Created {len(final_df)} hybrid logs")
    print(f"Final Learned Spend Mean: ${learned_spend_mean:.2f}")

if __name__ == "__main__":
    run_adaptive_loop(OUTPUT_TRAIN_FILE_NAME, OUTPUT_ADAPTIVE_TRAIN_FILE_NAME)