import numpy as np

def preprocess_state(row):
    is_weekend = 1.0 if row['isWeekend'] else 0.0
    time_of_day = float(row['timeOfDay']) / 24.0      # Normalize 24hr time
    occupancy = float(row['occupancy'])               # Already 0.0 to 1.0
    num_of_guests = float(row['numOfGuests']) / 8.0   # Assuming max 8 guests
    visit_count = float(row['customerVisitCount']) / 50.0 # Assuming max 50 visits
    avg_spend = float(row['customerAvgSpend']) / 100.0    # Approximate scaling

    state = np.array([
        is_weekend,
        time_of_day,
        occupancy,
        num_of_guests,
        visit_count,
        avg_spend
    ])
    
    # Reshape for Keras (1 sample, 6 features)
    return np.reshape(state, [1, 6])

def encode_action(action_type, assigned_capacity):
    # Maps action to discrete int, hard coded to the RESTAURANT_TABLES = [2, 2, 2, 4, 4, 4, 4, 6, 6, 8] for simplicity
    if action_type == 'reject' or assigned_capacity == 0:
        return 0
    elif assigned_capacity == 2: return 1
    elif assigned_capacity == 4: return 2
    elif assigned_capacity == 6: return 3
    elif assigned_capacity == 8: return 4
    
    return 0 # Default to reject if unknown