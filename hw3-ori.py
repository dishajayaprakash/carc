import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from jiwer import wer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def load_data(filename, start_index=None, end_index=None):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    if start_index is not None and end_index is not None:
        data = data.iloc[start_index:end_index]
        print(f"Data sliced from index {start_index} to {end_index}")
    print(f"Total samples loaded: {len(data)}")
    return data

class AutocorrectDataset(Dataset):
    def __init__(self, input_texts, target_texts, input_vocab, target_vocab, max_length=100):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_seq = [self.input_vocab['<sos>']] + [self.input_vocab.get(ch, self.input_vocab['<unk>']) for ch in self.input_texts[idx]] + [self.input_vocab['<eos>']]
        target_seq = [self.target_vocab['<sos>']] + [self.target_vocab.get(ch, self.target_vocab['<unk>']) for ch in self.target_texts[idx]] + [self.target_vocab['<eos>']]
        input_seq = input_seq[:self.max_length]
        target_seq = target_seq[:self.max_length]
        input_seq += [self.input_vocab['<pad>']] * (self.max_length - len(input_seq))
        target_seq += [self.target_vocab['<pad>']] * (self.max_length - len(target_seq))
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

def build_char_vocab(texts):
    chars = set()
    for text in texts:
        chars.update(text)
    vocab = {ch: idx + 4 for idx, ch in enumerate(sorted(chars))}
    vocab['<pad>'], vocab['<sos>'], vocab['<eos>'], vocab['<unk>'] = 0, 1, 2, 3
    return vocab

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, rnn_type='gru'):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))

    def forward(self, hidden, encoder_outputs):
        seq_len, batch_size, hid_dim = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs.transpose(0, 1)), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention=None, rnn_type='gru'):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.attention = attention
        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim + hid_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 2, output_dim) if attention else nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs=None):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        if self.attention is not None:
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            attn_weights = attn_weights.unsqueeze(1)
            context = torch.bmm(attn_weights, encoder_outputs.transpose(0, 1))
            context = context.transpose(0, 1)
            rnn_input = torch.cat((embedded, context), dim=2)
        else:
            rnn_input = embedded
        output, hidden = self.rnn(rnn_input, hidden)
        prediction = self.fc_out(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1)) if self.attention else self.fc_out(output.squeeze(0))
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.embedding.num_embeddings
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if np.random.rand() < teacher_forcing_ratio else top1
        return outputs

def train_seq2seq(model, iterator, optimizer, criterion, clip, epoch, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        src, trg = src.transpose(0, 1).to(model.device), trg.transpose(0, 1).to(model.device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def train_encoder(encoder, train_loader, optimizer, criterion):
    encoder.train()
    epoch_loss = 0
    for src, _ in train_loader:
        src = src.transpose(0, 1).to(encoder.device)
        optimizer.zero_grad()
        _, hidden = encoder(src)
        loss = criterion(hidden[-1], hidden[-1])  # Dummy loss for training demonstration
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def train_decoder(decoder, train_loader, optimizer, criterion):
    decoder.train()
    epoch_loss = 0
    for _, trg in train_loader:
        trg = trg.transpose(0, 1).to(decoder.device)
        optimizer.zero_grad()
        output, _ = decoder(trg[0], trg[1])
        loss = criterion(output, trg[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = load_data(args.train_filename, args.start_index, args.end_index)
    val_data = load_data(args.validation_filename)
    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()
    input_vocab = build_char_vocab(train_input_texts + val_input_texts)
    target_vocab = build_char_vocab(train_target_texts + val_target_texts)

    train_dataset = AutocorrectDataset(train_input_texts, train_target_texts, input_vocab, target_vocab, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.model_type == 'encoder':
        encoder = Encoder(len(input_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()  # Placeholder loss for training demonstration
        for epoch in range(args.num_epochs):
            train_loss = train_encoder(encoder, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Encoder Train Loss: {train_loss:.4f}")

    elif args.model_type == 'decoder':
        decoder = Decoder(len(target_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(args.num_epochs):
            train_loss = train_decoder(decoder, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Decoder Train Loss: {train_loss:.4f}")

    elif args.model_type == 'seq2seq':
        attention = Attention(args.hid_dim) if args.use_attention else None
        encoder = Encoder(len(input_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        decoder = Decoder(len(target_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attention=attention, rnn_type=args.rnn_type).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(args.num_epochs):
            train_loss = train_seq2seq(model, train_loader, optimizer, criterion, clip=1, epoch=epoch, teacher_forcing_ratio=args.teacher_forcing_ratio)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Seq2Seq Train Loss: {train_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq Training Script with Encoder/Decoder/Seq2Seq Options")
    parser.add_argument('--train_filename', type=str, required=True)
    parser.add_argument('--validation_filename', type=str, required=True)
    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--hid_dim', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_length', type=int, default=100)
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--rnn_type', type=str, choices=['rnn', 'gru'], default='gru')
    parser.add_argument('--model_type', type=str, choices=['encoder', 'decoder', 'seq2seq'], default='seq2seq')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
