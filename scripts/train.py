# scripts/train.py

import torch
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from model import UPIFraudGNN, get_loss

# ── Load graph ─────────────────────────────────────────────────────────────
data = torch.load("data/graph.pt", weights_only=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

# ── Model ──────────────────────────────────────────────────────────────────
model = UPIFraudGNN(hidden_dim=64, dropout=0.3).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# ── Move data to device ────────────────────────────────────────────────────
for node_type in data.node_types:
    data[node_type].x = data[node_type].x.to(device)
if hasattr(data['tx'], 'y'):
    data['tx'].y = data['tx'].y.to(device)

# Edge indices to device
for edge_type in data.edge_types:
    data[edge_type].edge_index = data[edge_type].edge_index.to(device)

# ── Training loop ──────────────────────────────────────────────────────────
def train_epoch():
    model.train()
    optimizer.zero_grad()
    
    logits, recon, _ = model(data.x_dict, data.edge_index_dict)
    
    mask   = data['tx'].train_mask
    labels = data['tx'].y[mask]
    
    loss = get_loss(
        logits[mask], labels,
        recon[mask],
        data['tx'].x[mask],
        lambda_=0.3
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(mask_name):
    model.eval()
    logits, _, _ = model(data.x_dict, data.edge_index_dict)
    
    mask   = getattr(data['tx'], mask_name)
    labels = data['tx'].y[mask].cpu().numpy()
    probs  = torch.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
    preds  = (probs > 0.5).astype(int)
    
    auc = roc_auc_score(labels, probs)
    return auc, labels, preds, probs


# ── Run training ───────────────────────────────────────────────────────────
best_val_auc = 0
patience = 15
patience_counter = 0

print("Epoch | Train Loss | Val AUC | Test AUC")
print("-" * 45)

for epoch in range(1, 201):
    loss = train_epoch()
    scheduler.step()
    
    if epoch % 5 == 0:
        val_auc,  _, _, _ = evaluate('val_mask')
        test_auc, y_true, y_pred, y_prob = evaluate('test_mask')
        
        print(f"  {epoch:3d} |   {loss:.4f}   |  {val_auc:.4f}  |  {test_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "models/best_gnn.pt")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# ── Final evaluation ───────────────────────────────────────────────────────
model.load_state_dict(torch.load("models/best_gnn.pt"))
_, y_true, y_pred, y_prob = evaluate('test_mask')

print("\n── Test Set Results ──")
print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")

# Save metadata for inference
import json
meta = {
    "hidden_dim": 64,
    "threshold": 0.5,
    "best_val_auc": best_val_auc
}
with open("models/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nModel saved to models/best_gnn.pt")