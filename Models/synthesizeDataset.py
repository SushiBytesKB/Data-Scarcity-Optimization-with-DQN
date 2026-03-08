import pandas as pd
import numpy as np
import random
import os

SEED = 30
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

GENERATE_TRAIN_ROWS = 100
GENERATE_TEST_ROWS = 1000
OUTPUT_TRAIN_FILE_NAME = 'Models/sparse_real_train.csv'
OUTPUT_TEST_FILE_NAME = 'Models/robust_real_test.csv'
RESTAURANT_TABLES = [2, 2, 2, 4, 4, 4, 4, 6, 6, 8] # Assuming max group size is 8 people
BASE_SPEND_PER_GUEST = 35.0 # Dollars
PRIME_TIME_HOURS = [12, 13, 18, 19]
HOURS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
HOUR_WEIGHTS = [
    0.05, 0.08, # Breakfast 
    0.1, 0.15, 0.15, # Lunch Rush 
    0.05, 0.02, # Afternoon
    0.1, 0.1, # Early Dinner
    0.15, 0.15, # Prime Dinner
    0.05, 0.04, 0.01 # Late Night
]

def generate_real_synthesis(num_rows, output_file_name):
    dataset = []
    print(f"Generating {num_rows} logs for file: {output_file_name}")
    
    for i in range(num_rows):
        # State Generation
        is_weekend = random.choice([True, False])
        time_of_day = random.choices(HOURS, weights=HOUR_WEIGHTS, k=1)[0]

        num_of_guests = random.choices([1, 2, 3, 4, 5, 6, 7, 8], weights=[0.05, 0.4, 0.05, 0.3, 0.05, 0.1, 0.02, 0.03])[0]
        customer_visit_count = random.choices(
            population=[0, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 50],
            weights=[0.4, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.04, 0.03, 0.02, 0.01]
        )[0]

        base_spend_dist = np.random.normal(30, 10) 
        visit_bonus = customer_visit_count * 0.5 
        customer_avg_spend = max(15, base_spend_dist + visit_bonus) 
        
        base_occ = 0.75 if time_of_day in PRIME_TIME_HOURS else 0.3
        if is_weekend: base_occ += 0.15
        occupancy = np.clip(base_occ + np.random.normal(0, 0.1), 0.0, 1.0)
        
        # Action Generation (Simulating past decisions)
        possible_tables = [t for t in RESTAURANT_TABLES if t >= num_of_guests]
        
        if not possible_tables:
            action_type = 'reject'
            assigned_table_capacity = 0
        else:
            rejection_chance = 0.5 if occupancy > 0.85 else 0.1
            if random.random() < rejection_chance:
                action_type = 'reject'
                assigned_table_capacity = 0
            else:
                action_type = 'accept'
                if random.random() < 0.7:
                    assigned_table_capacity = min(possible_tables) 
                else:
                    assigned_table_capacity = random.choice(possible_tables) 

        wasted_seats = max(0, assigned_table_capacity - num_of_guests)
        
        # Environment Dynamics (Duration)
        is_lunch = 11 <= time_of_day <= 14
        base_duration = 30 if is_lunch else 45
        duration = base_duration + (num_of_guests * 5) 
        if is_weekend: duration += 15
        actual_duration = max(30, duration + random.randint(-10, 15))

        # Reward / Target Q-Value Calculation
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

        target_q_value = price_paid - opportunity_cost

        dataset.append({
            'isWeekend': is_weekend,
            'timeOfDay': time_of_day,
            'occupancy': round(occupancy, 2),
            'numOfGuests': num_of_guests,
            'customerVisitCount': customer_visit_count,
            'customerAvgSpend': round(customer_avg_spend, 2),
            'actionType': action_type,
            'assignedTableCapacity': assigned_table_capacity,
            'wastedSeats': wasted_seats,
            'actualDuration': actual_duration,
            'targetQValue': round(target_q_value, 2) 
        })

    df = pd.DataFrame(dataset)
    
    # Save the file
    df.to_csv(output_file_name, index=False)
    print(f"Saved {output_file_name} successfully!\n")

if __name__ == "__main__":
    # Generate the Cold Start Dataset (Method 1 Baseline & Method 2 Adaptive Synthetic Training Loop)
    generate_real_synthesis(GENERATE_TRAIN_ROWS, OUTPUT_TRAIN_FILE_NAME)
    
    # Generate the Unseen Evaluation Dataset (For final comparison)
    generate_real_synthesis(GENERATE_TEST_ROWS, OUTPUT_TEST_FILE_NAME)