import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from models.efficientnet import get_efficientnet
from models.gat import GAT
from models.gru import GRU

class DeepfakeModel(nn.Module):
    def __init__(self, seq_len=40, dropout_rate=0.5):  # Increased dropout rate
        super(DeepfakeModel, self).__init__()
        self.seq_len = seq_len
        self.efficientnet = get_efficientnet()
        
        # Revised projection with dropout
        self.projection = nn.Sequential(
            nn.Linear(1280, 64),
            nn.GELU(),
            nn.Dropout(dropout_rate),  # New dropout layer
            nn.Linear(64, 4)
        )
        
        # GAT layer (assuming GAT implementation exists)
        self.gat = GAT(in_channels=4, out_channels=4, heads=1)
        
        # GRU layer with dropout
        self.gru = GRU(
            input_size=4, 
            hidden_size=16, 
            num_layers=1, 
            dropout=dropout_rate
        )
        
        # Attention layer with dropout
        self.attention = nn.Sequential(
            nn.Linear(16, 1),
            nn.Dropout(dropout_rate)  # New dropout layer
        )
        
        # Final fully connected layer
        self.fc = nn.Linear(16, 1)

    def forward(self, x, batched_edge_index):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)
        
        # EfficientNet feature extraction
        spatial_features = self.efficientnet(x).squeeze(-1).squeeze(-1)
        
        # Projection with dropout
        projected_features = self.projection(spatial_features)
        
        # GAT processing
        gat_output = self.gat(projected_features, batched_edge_index)
        gat_output = gat_output.view(batch_size, seq_len, -1)
        
        # GRU processing
        gru_output = self.gru(gat_output)
        
        # Attention mechanism with dropout
        attn_scores = self.attention(gru_output)
        attn_weights = torch.softmax(attn_scores, dim=1)
        weighted_output = torch.sum(gru_output * attn_weights, dim=1)
        
        # Final prediction
        output = self.fc(weighted_output)
        return output
