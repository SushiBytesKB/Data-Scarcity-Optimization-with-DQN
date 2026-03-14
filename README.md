# Data Scarcity in Multi-Tenant Systems: An Adaptive Synthetic Training Loop for Personalized Reinforcement Learning

By: Sushant Bharadwaj Kagolanu

Deep Reinforcement Learning (DRL) offers massive optimization potential for service industries, but deploying personalized DRL in multi-tenant Small and Medium-sized Enterprise (SME) environments is practically impossible due to the **"Cold Start" problem**. SMEs lack the massive historical datasets required to train complex neural networks on day one.

**This project solves data scarcity.** It introduces a novel **Adaptive Synthetic Training Loop** that uses a tiny fraction of real-world interactions (sparse data) to dynamically calibrate a data synthesizer. The synthesizer safely amplifies the dataset, allowing an Offline Deep Q-Network (DQN) to learn generalized environment policies without overfitting to noise.

Applied to a restaurant reservation management prototype (**ExquiTable**), this architecture successfully increased the model's simulated revenue generation by **over 150%** compared to standard sparse-data training.

---

## 🧠 Core Architecture & Academic Contribution

Rather than relying on massive historical datasets or naive random generation, this repository utilizes a mathematically sound, two-phase data synthesis pipeline:

1. **The Warm-Up Phase:** The synthesizer reads a small batch of initial ground-truth logs (20 interactions). Instead of generating fake data immediately, it uses an **Exponential Moving Average (EMA)** to safely anchor its generative parameters (e.g., expected customer spend, party size distribution) to the actual environment.
2. **Dynamic Amplification:** Once calibrated, the synthesizer dynamically generates mathematically sound transitions (State, Action, Reward, Next State) at a 10:1 synthetic-to-real ratio.
3. **Deterministic Evaluation:** Synthetic states are passed through a deterministic environment simulator to calculate exact Opportunity Costs and Rewards, ensuring the RL agent learns valid logic.

---

## 📊 Experimental Results

To prove the hypothesis, the experiment was built with strict scientific controls: Global random seeds for reproducibility, proportional batch-sizing to ensure equal gradient updates (10 steps per epoch), and Keras Early Stopping to prevent memorization.

* **Baseline Model (Trained strictly on 100 sparse logs):**
* Memorized the noise of the sparse dataset (Low Training MSE).
* Triggered Early Stopping prematurely at ~50 epochs.
* Failed to learn capacity rules (e.g., seating 6 people at a 2-top table), resulting in heavy penalties and a final cumulative reward of **<$20,000**.


* **Adaptive Model (Trained on 100 real + 960 synthetic logs):**
* Maintained a generalized understanding of the environment, surviving to ~95 epochs before Early Stopping.
* Successfully learned complex Opportunity Cost logic (e.g., rejecting low-value walk-ins during prime-time rushes to save tables for highly profitable reservations).
* Achieved a final cumulative reward of **~$50,000**.

---

## ⚙️ Tech Stack & Engineering Practices

* **Algorithm:** Offline (Batch) Deep Q-Learning (DQN).
* **Neural Network Architecture:** Shallow, dense feed-forward network optimized for tabular continuous/discrete state-action mappings (TensorFlow/Keras).
* **Data Engineering:** Pandas & NumPy for vectorized state normalization and EMA calculations.
* **Deployment:** Headless Flask REST API, fully decoupled from database infrastructure for easy local reproducibility and microservice integration.
* **MLOps:** Automated training pipelines, deterministic seeding, and dynamic performance visualization (Matplotlib).

---

## 🚀 Local Installation & Usage

This repository is designed to be fully reproducible locally without the need for external database connections.

### 1. Set up the Environment

```bash
git clone https://github.com/SushiBytesKB/Data-Scarcity-Optimization-with-DQN.git
cd Data-Scarcity-Optimization-with-DQN
pip install -r requirements.txt

```

### 2. Run the Offline Training Pipeline

This script will generate the deterministic ground-truth data, run the Adaptive Synthesizer, and train both the Baseline and Adaptive models side-by-side.

```bash
python Models/synthesizeDataset.py
python Models/adaptiveTrainingLoop.py
python trainModels.py

```

### 3. Evaluate and Visualize

Generate the comparative analysis metrics and output the training loss and cumulative reward graphs (saved as `.png` files).

```bash
python evaluateModels.py

```

### 4. Spin Up the Headless Inference API

Launch the Flask server to interact with the winning Adaptive RL agent in real-time.

```bash
python server.py

```

Test the API by sending a mock customer walk-in state to the local endpoint:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
    "isWeekend": true,
    "timeOfDay": 19,
    "occupancy": 0.85,
    "numOfGuests": 4,
    "customerVisitCount": 2,
    "customerAvgSpend": 45.50
}'

```

*The API will return a JSON payload detailing the optimal action (Accept/Reject), the ideal table assignment, and the predicted Q-values for all possible decisions.*

---

## 📂 Repository Structure

```text
├── agent.py                  # Core DQNAgent class (Keras NN build, offline training logic)
├── preprocessor.py           # State vector normalization and action decoding
├── server.py                 # Flask REST API for model inference
├── trainModels.py            # Automated training orchestrator (Early Stopping, Checkpointing)
├── evaluateModels.py         # Comparative analysis and Matplotlib graph generation
├── requirements.txt          # Minimal dependency requirements
└── Models/
    ├── synthesizeDataset.py      # Hidden ruleset simulating the restaurant
    └── adaptiveTrainingLoop.py   # EMA parameter tracking & dynamic data synthesis

```

---

## 🔭 Future Work

* **Online Reinforcement Learning:** Transitioning the offline DQN into an online learner that updates its neural weights via nightly micro-batches as genuine live customer data flows in.
* **Generative Architectures:** Upgrading the linear EMA synthesizer to utilize Variational Autoencoders (VAEs) to capture highly complex, non-linear correlations in sparse data.
* **Different Environments:** Applying the adaptive training loop to other business sectors and systems will provide an inference regarding the consistency of the methodology.

---

*This project was developed and presented for the TUJ Academic Conference 2026.*
