# scripts/export_onnx.py

import torch
import json
from model import UPIFraudGNN

model = UPIFraudGNN(hidden_dim=64)
model.load_state_dict(torch.load("models/best_gnn.pt", map_location='cpu'))
model.eval()

# For ONNX export we need a simplified scoring-only version
# that takes pre-aggregated node features as input
# (the GNN aggregation happens at scoring time in Java via a simpler scoring approach)

# Easier approach: export just the CLASSIFIER HEAD
# We'll compute GNN embeddings in Python, cache them, 
# and only run the final MLP in Java

class ClassifierOnly(torch.nn.Module):
    def __init__(self, gnn):
        super().__init__()
        self.classifier = gnn.classifier
    
    def forward(self, tx_embedding):
        return self.classifier(tx_embedding)

clf = ClassifierOnly(model)
dummy_input = torch.randn(1, 64)

torch.onnx.export(
    clf,
    dummy_input,
    "models/fraud_classifier.onnx",
    input_names=["tx_embedding"],
    output_names=["fraud_logits"],
    dynamic_axes={"tx_embedding": {0: "batch_size"}}
)
print("ONNX model exported to models/fraud_classifier.onnx")