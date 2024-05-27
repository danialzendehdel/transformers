import torch.nn as nn
from tranformer.Encoder.attention import MultiHeadAttention
from tranformer.Encoder.feedforward import FeedForwardNetwork, AddNorm
import torch

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, ffn_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.add_norm1 = AddNorm(model_dim)
        self.encoder_decoder_attention = MultiHeadAttention(model_dim, num_heads)
        self.add_norm2 = AddNorm(model_dim)
        self.feed_forward = FeedForwardNetwork(model_dim, ffn_dim)
        self.add_norm3 = AddNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        # Self-attention
        self_attention_output, _ = self.self_attention(x, x, x, look_ahead_mask)
        self_attention_output = self.dropout(self_attention_output)
        x = self.add_norm1(x, self_attention_output)

        # Encoder-decoder attention
        encoder_decoder_attention_output, _ = self.encoder_decoder_attention(x, encoder_output, encoder_output,
                                                                             padding_mask)
        encoder_decoder_attention_output = self.dropout(encoder_decoder_attention_output)
        x = self.add_norm2(x, encoder_decoder_attention_output)

        # Feed-forward network
        ffn_output = self.feed_forward(x)
        ffn_output = self.dropout(ffn_output)
        x = self.add_norm3(x, ffn_output)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads, ffn_dim, target_vocab_size, max_seq_len, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(target_vocab_size, model_dim)
        self.positional_encoding = self.create_positional_encoding(max_seq_len, model_dim)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(model_dim, num_heads, ffn_dim, dropout_rate) for _ in range(num_layers)
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

    def forward(self, x, encoder_output, look_ahead_mask=None, padding_mask=None):
        seq_len = x.size(1)
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.model_dim, dtype=torch.float32))
        x = x + self.positional_encoding[:seq_len, :].transpose(0, 1)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)

        return x

