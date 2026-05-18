# scripts/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class UPIFraudGNN(nn.Module):
    """
    Heterogeneous GNN for UPI fraud detection using HeteroConv + SAGEConv.
    """
    def __init__(self, hidden_dim=64, dropout=0.3):
        super().__init__()
        
        # Project each node type into shared hidden space
        self.user_proj   = Linear(4, hidden_dim)
        self.device_proj = Linear(2, hidden_dim)
        self.tx_proj     = Linear(5, hidden_dim)
        
        # Heterogeneous convolution layers (using SAGEConv)
        self.conv1 = HeteroConv({
            ('user',   'sent',        'tx'):     SAGEConv((-1, -1), hidden_dim),
            ('tx',     'received_by', 'user'):   SAGEConv((-1, -1), hidden_dim),
            ('user',   'uses',        'device'): SAGEConv((-1, -1), hidden_dim),
            ('device', 'carried',     'tx'):     SAGEConv((-1, -1), hidden_dim),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('user',   'sent',        'tx'):     SAGEConv(hidden_dim, hidden_dim),
            ('tx',     'received_by', 'user'):   SAGEConv(hidden_dim, hidden_dim),
            ('user',   'uses',        'device'): SAGEConv(hidden_dim, hidden_dim),
            ('device', 'carried',     'tx'):     SAGEConv(hidden_dim, hidden_dim),
        }, aggr='sum')
        
        # Classifier head for transactions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Decoder for reconstruction loss (unsupervised component)
        self.decoder = nn.Linear(hidden_dim, 5)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # 1. Project features
        h = {
            'user':   F.relu(self.user_proj(x_dict['user'])),
            'device': F.relu(self.device_proj(x_dict['device'])),
            'tx':     F.relu(self.tx_proj(x_dict['tx'])),
        }
        
        # 2. First GNN layer
        h = self.conv1(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: self.dropout(v) for k, v in h.items()}
        
        # 3. Second GNN layer
        h = self.conv2(h, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        
        # 4. Transaction output
        tx_emb = h['tx']
        logits = self.classifier(tx_emb)
        recon  = self.decoder(tx_emb)
        
        return logits, recon, tx_emb


def get_loss(logits, labels, recon, original_features, lambda_=0.3):
    weights = torch.tensor([1.0, 33.0], device=logits.device)
    ce_loss = F.cross_entropy(logits, labels, weight=weights)
    recon_loss = F.mse_loss(recon, original_features)
    return lambda_ * ce_loss + (1 - lambda_) * recon_loss