import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from preprocessor import preprocess_state, encode_action

class DQNAgent:
    def __init__(self, state_size=6, action_size=5):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Build Neural Network for the Deep Q-Network
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        # Linear activation for the output because Q-values can be negative or positive
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def train_from_csv(self, csv_path, epochs=50, batch_size=32, validation_split=0.2):
        # Offline RL Training Loop
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Pre-calculate all state vectors
        states = np.array([preprocess_state(row)[0] for _, row in df.iterrows()])
        
        # Batch predict to get current Q-value estimates for ALL actions
        print("Predicting baseline Q-values")
        targets = self.model.predict(states, batch_size=batch_size, verbose=0)

        # Update the specific target for the action taken in the log
        for i, row in df.iterrows():
            action_idx = encode_action(row['actionType'], row['assignedTableCapacity'])
            reward = row['targetQValue']
            
            # Overwrite the Q-value for the taken action with the known reward
            targets[i][action_idx] = reward

        # Train the model to map the state to the updated Q-values
        early_stop = EarlyStopping(
            monitor='val_loss', 
            patience=10, 
            restore_best_weights=True, 
            verbose=1
        )

        print(f"Training model on {len(states)} samples (80% Train / 20% Val)...")
        history = self.model.fit(
            states, targets, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1
        )
        print("Training complete.\n")
        
        # Return the history dictionary to save for graphs
        return history.history

    def save(self, name):
        self.model.save(name)
        print(f"Saved model as {name}")

    def load(self, name):
        if os.path.exists(name):
            self.model.load_weights(name)
            print(f"Loaded weights from {name}")
        else:
            print(f"Could not find {name}. Starting with random weights.")