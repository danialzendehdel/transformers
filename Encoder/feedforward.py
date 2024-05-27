import torch
import torch.nn as nn
import torch.nn.functional as F
from tranformer.Encoder.attention import MultiHeadAttention


# Feed-Forward Network
class FeedForwardNetwork(nn.Module):
    def __init__(self, model_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(model_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, model_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Add & Norm
class AddNorm(nn.Module):
    def __init__(self, model_dim, epsilon=1e-6):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim, eps=epsilon)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)


# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(model_dim, num_heads)
        self.add_norm1 = AddNorm(model_dim)
        self.feed_forward = FeedForwardNetwork(model_dim, ffn_dim)
        self.add_norm2 = AddNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        attn_output = self.dropout(attn_output)
        x = self.add_norm1(x, attn_output)

        # Feed-forward network
        ffn_output = self.feed_forward(x)
        ffn_output = self.dropout(ffn_output)
        x = self.add_norm2(x, ffn_output)

        return x

