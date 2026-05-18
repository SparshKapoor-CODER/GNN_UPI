# scripts/build_graph.py

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv("data/transactions.csv")

# ── Node encoders ─────────────────────────────────────────────────────────────
user_enc   = LabelEncoder().fit(list(set(df['sender'].tolist() + df['receiver'].tolist())))
device_enc = LabelEncoder().fit(df['device'].tolist())
tx_enc     = LabelEncoder().fit(df['tx_id'].tolist())

df['sender_id']   = user_enc.transform(df['sender'])
df['receiver_id'] = user_enc.transform(df['receiver'])
df['device_id']   = device_enc.transform(df['device'])
df['tx_idx']      = tx_enc.transform(df['tx_id'])

num_users   = len(user_enc.classes_)
num_devices = len(device_enc.classes_)
num_txns    = len(df)

# ── Node features ─────────────────────────────────────────────────────────────
# User features: avg sent amount, avg received amount, txn count
user_features = np.zeros((num_users, 4))
for _, row in df.iterrows():
    s, r = row['sender_id'], row['receiver_id']
    user_features[s][0] += row['amount_normalized']  # total sent (normalized)
    user_features[s][2] += 1                          # sent count
    user_features[r][1] += row['amount_normalized']  # total received
    user_features[r][3] += 1                          # received count

# Normalize counts
user_features[:, 2] /= (user_features[:, 2].max() + 1e-8)
user_features[:, 3] /= (user_features[:, 3].max() + 1e-8)

# Device features: number of distinct users, avg hop count
device_user_count = np.zeros(num_devices)
device_hop_sum    = np.zeros(num_devices)
device_tx_count   = np.zeros(num_devices)
for _, row in df.iterrows():
    d = row['device_id']
    device_tx_count[d] += 1
    device_hop_sum[d]  += row['hop_count']

device_features = np.stack([
    device_tx_count / (device_tx_count.max() + 1e-8),
    device_hop_sum  / (device_hop_sum.max()  + 1e-8),
], axis=1)

# Transaction features
tx_features = df[[
    'amount_normalized',
    'hour_of_day',
    'hop_count',
    'velocity_10min',
    'is_new_pair'
]].values.astype(np.float32)

# Normalize
tx_features[:, 1] /= 24.0   # hour → [0,1]
tx_features[:, 2] /= 8.0    # hop  → [0,1]
tx_features[:, 3] /= 20.0   # velocity → [0,1]

# ── Build HeteroData ──────────────────────────────────────────────────────────
data = HeteroData()

data['user'].x   = torch.tensor(user_features,  dtype=torch.float)
data['device'].x = torch.tensor(device_features, dtype=torch.float)
data['tx'].x     = torch.tensor(tx_features,    dtype=torch.float)

# Labels (only on transaction nodes)
data['tx'].y = torch.tensor(df['is_fraud'].values, dtype=torch.long)

# Edge: sender --sent--> tx
data['user', 'sent', 'tx'].edge_index = torch.tensor(
    [df['sender_id'].values, df['tx_idx'].values], dtype=torch.long)

# Edge: tx --received_by--> user
data['tx', 'received_by', 'user'].edge_index = torch.tensor(
    [df['tx_idx'].values, df['receiver_id'].values], dtype=torch.long)

# Edge: user --uses--> device
data['user', 'uses', 'device'].edge_index = torch.tensor(
    [df['sender_id'].values, df['device_id'].values], dtype=torch.long)

# Edge: device --carried--> tx
data['device', 'carried', 'tx'].edge_index = torch.tensor(
    [df['device_id'].values, df['tx_idx'].values], dtype=torch.long)

# Train/val/test split (chronological — no leakage)
n = num_txns
train_mask = torch.zeros(n, dtype=torch.bool)
val_mask   = torch.zeros(n, dtype=torch.bool)
test_mask  = torch.zeros(n, dtype=torch.bool)

train_mask[:int(0.6*n)]                    = True
val_mask[int(0.6*n):int(0.8*n)]           = True
test_mask[int(0.8*n):]                     = True

data['tx'].train_mask = train_mask
data['tx'].val_mask   = val_mask
data['tx'].test_mask  = test_mask

torch.save(data, "data/graph.pt")
print("Graph saved.")
print(f"  Users: {num_users}, Devices: {num_devices}, Txns: {num_txns}")
print(f"  Fraud rate: {df['is_fraud'].mean()*100:.1f}%")