from flask import Flask, request, jsonify
import numpy as np
from agent import DQNAgent
from preprocessor import preprocess_state

app = Flask(__name__)

print("Initializing API, loading Adaptive Model")
agent = DQNAgent()

# Method 2 (Adaptive) model as the default
MODEL_PATH = 'Models/adaptive_dqn.h5'
try:
    agent.load(MODEL_PATH)
    print(f"Successfully loaded {MODEL_PATH} for inference")
except Exception as e:
    print(f"Error: {e}")

# Action decoder mapping (The inverse of encode_action in preprocessor.py)
def decode_action(action_idx):
    action_map = {
        0: {"action": "reject", "table_capacity": 0},
        1: {"action": "accept", "table_capacity": 2},
        2: {"action": "accept", "table_capacity": 4},
        3: {"action": "accept", "table_capacity": 6},
        4: {"action": "accept", "table_capacity": 8}
    }
    return action_map.get(action_idx, {"action": "reject", "table_capacity": 0})

# Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running."}), 200

@app.route('/predict', methods=['POST'])
def predict_action():
    try:
        # Parse the incoming JSON request
        data = request.get_json()
        
        required_keys = ['isWeekend', 'timeOfDay', 'occupancy', 'numOfGuests', 'customerVisitCount', 'customerAvgSpend']
        if not data or not all(key in data for key in required_keys):
            return jsonify({"error": f"Missing required fields. Expected: {required_keys}"}), 400

        # Preprocess the state 
        state_vector = preprocess_state(data)

        q_values = agent.model.predict(state_vector, verbose=0)[0]
        
        # Select the best action (Highest Q-Value)
        best_action_idx = np.argmax(q_values)
        best_action = decode_action(best_action_idx)

        response = {
            "status": "success",
            "input_state": data,
            "predicted_q_values_array": [round(float(q), 2) for q in q_values],
            "recommended_action": best_action['action'],
            "recommended_table": best_action['table_capacity']
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting server on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False)