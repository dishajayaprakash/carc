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
from torch.cuda.amp import GradScaler, autocast
import torch.utils.checkpoint as checkpoint
import gc

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
        outputs, hidden = checkpoint.checkpoint(self.rnn, embedded)
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
        output, hidden = checkpoint.checkpoint(self.rnn, rnn_input, hidden)
        if self.attention is not None:
            prediction = self.fc_out(torch.cat((output.squeeze(0), context.squeeze(0)), dim=1))
        else:
            prediction = self.fc_out(output.squeeze(0))
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

def train_seq2seq(model, iterator, optimizer, criterion, clip, epoch, teacher_forcing_ratio, accumulation_steps):
    model.train()
    epoch_loss = 0
    scaler = GradScaler()
    optimizer.zero_grad()
    print(f"Starting training for epoch {epoch+1}...")
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.transpose(0, 1).to(model.device), trg.transpose(0, 1).to(model.device)
        with autocast():
            output = model(src, trg, teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg) / accumulation_steps
        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        epoch_loss += loss.item() * accumulation_steps
        # Clear cache and collect garbage periodically
        if (i + 1) % (100 * accumulation_steps) == 0:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Epoch [{epoch+1}], Batch [{i+1}/{len(iterator)}], Loss: {loss.item() * accumulation_steps:.4f}")
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, target_vocab, epoch):
    model.eval()
    epoch_loss = 0
    wers = []
    inv_target_vocab = {idx: ch for ch, idx in target_vocab.items()}
    scaler = GradScaler()
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.transpose(0, 1).to(model.device), trg.transpose(0, 1).to(model.device)
            with autocast():
                output = model(src, trg, 0)  # No teacher forcing during evaluation
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].reshape(-1)
                loss = criterion(output, trg)
            epoch_loss += loss.item()
            preds = output.argmax(1).reshape(-1)
            trg = trg.reshape(-1)
            for pred, target in zip(preds.cpu().numpy(), trg.cpu().numpy()):
                pred_char = inv_target_vocab.get(pred, '<unk>')
                trg_char = inv_target_vocab.get(target, '<unk>')
                if trg_char in ['<pad>', '<sos>', '<eos>']:
                    continue
                if pred_char in ['<pad>', '<sos>', '<eos>']:
                    pred_char = '<unk>'
                wer_score = wer(trg_char, pred_char)
                wers.append(wer_score)
            # Clear cache and collect garbage periodically
            if (i + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Evaluation Batch [{i+1}/{len(iterator)}], Current Loss: {loss.item():.4f}")
    avg_loss = epoch_loss / len(iterator)
    avg_wer = sum(wers) / len(wers) if len(wers) > 0 else 0
    print(f"Epoch [{epoch+1}] Evaluation Completed. Average Loss: {avg_loss:.4f}, Average WER: {avg_wer:.4f}")
    return avg_loss, avg_wer

def build_positional_encoding(emb_dim, dropout, max_len=5000):
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, emb_dim, 2) * -(np.log(10000.0) / emb_dim))
    pe = torch.zeros(max_len, 1, emb_dim)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    pe = pe.transpose(0, 1)
    return nn.Parameter(pe, requires_grad=False)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = build_positional_encoding(emb_dim, dropout, max_len)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    train_data = load_data(args.train_filename, args.start_index, args.end_index)
    val_data = load_data(args.validation_filename)
    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()
    input_vocab = build_char_vocab(train_input_texts + val_input_texts)
    target_vocab = build_char_vocab(train_target_texts + val_target_texts)

    train_dataset = AutocorrectDataset(train_input_texts, train_target_texts, input_vocab, target_vocab, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = AutocorrectDataset(val_input_texts, val_target_texts, input_vocab, target_vocab, args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if args.model_type == 'encoder':
        encoder = Encoder(len(input_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.MSELoss()  # Dummy loss for training demonstration
        for epoch in range(args.num_epochs):
            train_loss = train_encoder(encoder, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Encoder Train Loss: {train_loss:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

    elif args.model_type == 'decoder':
        decoder = Decoder(len(target_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        for epoch in range(args.num_epochs):
            train_loss = train_decoder(decoder, train_loader, optimizer, criterion)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Decoder Train Loss: {train_loss:.4f}")
            torch.cuda.empty_cache()
            gc.collect()

    elif args.model_type == 'seq2seq':
        attention = Attention(args.hid_dim) if args.use_attention else None
        encoder = Encoder(len(input_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, rnn_type=args.rnn_type).to(device)
        decoder = Decoder(len(target_vocab), args.emb_dim, args.hid_dim, args.n_layers, args.dropout, attention=attention, rnn_type=args.rnn_type).to(device)
        model = Seq2Seq(encoder, decoder, device).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        best_val_wer = float('inf')
        patience_counter = 0
        for epoch in range(args.num_epochs):
            train_loss = train_seq2seq(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                clip=1, 
                epoch=epoch, 
                teacher_forcing_ratio=args.teacher_forcing_ratio,
                accumulation_steps=args.accumulation_steps
            )
            val_loss, val_wer = evaluate(model, val_loader, criterion, target_vocab, epoch=epoch)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")

            # Save the best model based on WER
            if val_wer < best_val_wer:
                best_val_wer = val_wer
                torch.save(model.state_dict(), 'best_seq2seq_transformer_model.pth')
                print(f"Best model saved with Val WER: {best_val_wer:.4f}")
                patience_counter = 0  # Reset counter if performance improves
            else:
                patience_counter += 1
                print(f"No improvement in Val WER. Patience counter: {patience_counter}/{args.patience}")
                if args.use_early_stopping and patience_counter >= args.patience:
                    print("Early stopping triggered.")
                    break  # Early stopping

            # Clear GPU cache and collect garbage after each epoch
            torch.cuda.empty_cache()
            gc.collect()

        # Save the final model and vocabularies
        print("Training completed. Saving final model and vocabularies...")
        torch.save(model.state_dict(), 'transformer_seq2seq_model.pth')
        with open('input_vocab.pkl', 'wb') as f:
            pickle.dump(input_vocab, f)
        with open('target_vocab.pkl', 'wb') as f:
            pickle.dump(target_vocab, f)
        print("Model and vocabularies saved.")

        # Optionally, perform final evaluation on the validation set
        print("\n=== Final Evaluation on Validation Set ===")
        wers = []
        corrected_texts = []
        for idx, (src, trg) in enumerate(tqdm(val_loader, total=len(val_loader))):
            src, trg = src.transpose(0, 1).to(device), trg.transpose(0, 1).to(device)
            with torch.no_grad():
                with autocast():
                    output = model(src, trg, 0)  # No teacher forcing
            preds = output.argmax(2)  # [max_len, batch_size]
            for i in range(preds.shape[1]):
                pred_seq = preds[:, i].cpu().numpy()
                trg_seq = trg[:, i].cpu().numpy()
                pred_chars = [target_vocab.get(idx, '<unk>') for idx in pred_seq]
                trg_chars = [target_vocab.get(idx, '<unk>') for idx in trg_seq]
                pred_text = ''.join([ch for ch in pred_chars if ch not in ['<pad>', '<sos>', '<eos>']])
                trg_text = ''.join([ch for ch in trg_chars if ch not in ['<pad>', '<sos>', '<eos>']])
                wer_score = wer(trg_text, pred_text)
                wers.append(wer_score)
            # Clear cache and collect garbage periodically
            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Processed {idx+1}/{len(val_loader)} batches. Current WER: {wer_score:.4f}")
        average_wer = sum(wers) / len(wers) if len(wers) > 0 else 0
        print(f'Final Validation Average WER: {average_wer:.4f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq Training Script with Encoder/Decoder/Seq2Seq Options")
    parser.add_argument('--train_filename', type=str, required=True, help='Path to training data CSV file')
    parser.add_argument('--validation_filename', type=str, required=True, help='Path to validation data CSV file')
    parser.add_argument('--start_index', type=int, default=None, help='Start index for training data slicing')
    parser.add_argument('--end_index', type=int, default=None, help='End index for training data slicing')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay coefficient')
    parser.add_argument('--emb_dim', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dimension size for Transformer feedforward layers')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in the Transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=60, help='Maximum sequence length for padding')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism in the decoder')
    parser.add_argument('--rnn_type', type=str, choices=['rnn', 'gru'], default='gru', help='Type of RNN to use (rnn or gru)')
    parser.add_argument('--model_type', type=str, choices=['encoder', 'decoder', 'seq2seq'], default='seq2seq', help='Type of model to train')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='Probability to use teacher forcing')
    parser.add_argument('--use_weight_decay', action='store_true', help='Use weight decay (L2 regularization)')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing in the loss function')
    parser.add_argument('--label_smoothing_value', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping based on validation WER')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--use_gradient_clipping', action='store_true', help='Use gradient clipping during training')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Maximum norm for gradient clipping')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps')
    args = parser.parse_args()
    main(args)
