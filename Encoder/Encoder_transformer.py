import torch
import torch.nn as nn
from tranformer.Encoder.feedforward import EncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, input_vocab_size, max_seq_len, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_vocab_size, model_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, model_dim)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(model_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)

    def create_positional_encoding(self, max_seq_len, model_dim):
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def forward(self, x, mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x = x + self.positional_encoding[:seq_len, :].transpose(0, 1)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x
