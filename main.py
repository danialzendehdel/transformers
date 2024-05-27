import torch.nn as nn
import torch

from tranformer.Encoder.Encoder_transformer import TransformerEncoder
from tranformer.Decoder.decoder import TransformerDecoder
import torch.optim as optim

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader, TensorDataset


class TransformerBinaryClassifier(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, model_dim, num_heads, ffn_dim, input_vocab_size,
                 target_vocab_size, max_seq_len, num_classes=2, dropout_rate=0.1):
        super(TransformerBinaryClassifier, self).__init__()
        self.encoder = TransformerEncoder(num_encoder_layers, model_dim, num_heads, ffn_dim, input_vocab_size,
                                          max_seq_len, dropout_rate)
        self.decoder = TransformerDecoder(num_decoder_layers, model_dim, num_heads, ffn_dim, target_vocab_size,
                                          max_seq_len, dropout_rate)
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        # Taking the mean of the decoder output across the sequence length dimension
        decoder_output = decoder_output.mean(dim=1)
        output = self.fc(decoder_output)
        return output


# Load the IMDB dataset
train_iter, test_iter = IMDB(split=('train', 'test'))

# Tokenizer and vocabulary
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, [label + ' ' + line for (label, line) in train_iter]),
                                  specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


def preprocess(text, vocab, tokenizer, seq_len):
    tokens = tokenizer(text)
    tokens = tokens[:seq_len]  # truncate to max length
    tokens += ["<pad>"] * (seq_len - len(tokens))  # pad to max length
    return torch.tensor(vocab(tokens), dtype=torch.long)


seq_len = 200

# Preprocess the train and test datasets
train_dataset = [(preprocess(line, vocab, tokenizer, seq_len), 1 if label == 'pos' else 0) for (label, line) in
                 to_map_style_dataset(train_iter)]
test_dataset = [(preprocess(line, vocab, tokenizer, seq_len), 1 if label == 'pos' else 0) for (label, line) in
                to_map_style_dataset(test_iter)]

batch_size = 32

train_data = DataLoader(TensorDataset(*zip(*train_dataset)), batch_size=batch_size, shuffle=True)
test_data = DataLoader(TensorDataset(*zip(*test_dataset)), batch_size=batch_size)

# Model parameters
model_dim = 32
num_heads = 4
ffn_dim = 64
num_encoder_layers = 2
num_decoder_layers = 2
max_seq_len = seq_len
num_classes = 2

input_vocab_size = len(vocab)
target_vocab_size = len(vocab)

model = TransformerBinaryClassifier(num_encoder_layers, num_decoder_layers, model_dim, num_heads, ffn_dim,
                                    input_vocab_size, target_vocab_size, max_seq_len, num_classes)

# Loss and optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training function
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src[0].to(device), tgt.to(device)  # Extract tensors from TensorDataset
        optimizer.zero_grad()
        output = model(src, src)  # Using src as tgt for this demo
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src[0].to(device), tgt.to(device)  # Extract tensors from TensorDataset
            output = model(src, src)  # Using src as tgt for this demo
            loss = criterion(output, tgt)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == tgt).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy


# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
# Training and evaluation loop
for epoch in range(num_epochs):
    train_loss = train(model, train_data, criterion, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_data, criterion, device)
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
