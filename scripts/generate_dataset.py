# scripts/generate_dataset.py

import json
import random
import uuid
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

random.seed(42)
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────────
NUM_USERS = 200
NUM_DEVICES = 180         # some users share devices (fraud indicator)
NUM_TRANSACTIONS = 10000
FRAUD_RATIO = 0.03        # 3% fraud, like real UPI data

# ── Generate users ───────────────────────────────────────────────────────────
users = [f"user_{i}@demo" for i in range(NUM_USERS)]
user_balances = {u: random.uniform(500, 50000) for u in users}

# ── Generate devices ─────────────────────────────────────────────────────────
# Most users have their own device; ~10% of devices are "shared" (fraud pattern)
devices = [f"device_{i}" for i in range(NUM_DEVICES)]
user_to_device = {}
for i, u in enumerate(users):
    if i < NUM_DEVICES:
        user_to_device[u] = devices[i]
    else:
        # This user shares someone else's device - suspicious
        user_to_device[u] = random.choice(devices[:20])

# ── Generate transactions ────────────────────────────────────────────────────
transactions = []

# Normal transaction patterns
normal_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
fraud_hours  = [0, 1, 2, 3, 4, 23]  # late night = higher fraud

base_time = datetime(2024, 1, 1)

for i in range(NUM_TRANSACTIONS):
    is_fraud = random.random() < FRAUD_RATIO
    
    sender = random.choice(users)
    # Fraud: sender targets specific high-balance receivers
    if is_fraud:
        receiver = max(random.sample(users, 5), key=lambda u: user_balances[u])
    else:
        receiver = random.choice([u for u in users if u != sender])
    
    # Fraud patterns (mirroring what Penaganti paper describes):
    # 1. Unusual hour
    # 2. High velocity (many txns in short window)
    # 3. Amount slightly below round-number detection thresholds
    # 4. Device shared across multiple accounts
    # 5. New sender-receiver pair with large amount
    
    if is_fraud:
        hour = random.choice(fraud_hours)
        amount = random.uniform(9800, 9999)  # just under ₹10k threshold
        hop_count = random.randint(4, 8)     # many hops = more suspicious
    else:
        hour = random.choice(normal_hours)
        amount = random.uniform(10, 5000)
        hop_count = random.randint(1, 4)
    
    timestamp = base_time + timedelta(
        days=random.randint(0, 180),
        hours=hour,
        minutes=random.randint(0, 59)
    )
    
    # Velocity: count how many txns this sender did in last 10 minutes
    # (simplified: use random for dataset generation)
    velocity = random.randint(8, 20) if is_fraud else random.randint(0, 3)
    
    tx = {
        "tx_id": str(uuid.uuid4()),
        "sender": sender,
        "receiver": receiver,
        "device": user_to_device[sender],
        "amount": round(amount, 2),
        "timestamp": timestamp.isoformat(),
        "hour_of_day": hour,
        "hop_count": hop_count,
        "velocity_10min": velocity,
        "amount_normalized": amount / 10000,
        "is_new_pair": random.random() < (0.8 if is_fraud else 0.1),
        "is_fraud": int(is_fraud)
    }
    transactions.append(tx)

# Save
df = pd.DataFrame(transactions)
df.to_csv("data/transactions.csv", index=False)
print(f"Generated {len(df)} transactions, {df['is_fraud'].sum()} fraud ({df['is_fraud'].mean()*100:.1f}%)")