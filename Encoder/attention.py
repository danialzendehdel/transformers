import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Attention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0, "model_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.model_dim = model_dim
        self.depth = model_dim // num_heads

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)

        self.dense = nn.Linear(model_dim, model_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, depth)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        matmul_qk = torch.matmul(Q, K.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len_q, seq_len_k)
        dk = K.size(-1)
        scaled_attention_logits = matmul_qk / (dk ** 0.5)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len_q, depth_v)

        return output, attention_weights

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.wq(Q)
        K = self.wk(K)
        V = self.wv(V)

        Q = self.split_heads(Q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_heads(K, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        V = self.split_heads(V, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        scaled_attention = scaled_attention.permute(0, 2, 1, 3).contiguous()
        original_size_attention = scaled_attention.view(batch_size, -1,
                                                        self.model_dim)  # (batch_size, seq_len_q, model_dim)

        output = self.dense(original_size_attention)  # (batch_size, seq_len_q, model_dim)

        return output, attention_weights
