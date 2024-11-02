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

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Modified Dataset Class (Renamed for Clarity)
class TextCorrectionDataset(Dataset):
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

        # Truncate sequences if they exceed max_length
        input_seq = input_seq[:self.max_length]
        target_seq = target_seq[:self.max_length]

        # Padding
        input_seq += [self.input_vocab['<pad>']] * (self.max_length - len(input_seq))
        target_seq += [self.target_vocab['<pad>']] * (self.max_length - len(target_seq))

        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        return input_seq, target_seq

# Vocabulary Building Function
def build_char_vocab(texts):
    print("Building character vocabulary...")
    chars = set()
    for text in texts:
        chars.update(text)
    vocab = {ch: idx + 4 for idx, ch in enumerate(sorted(chars))}
    vocab['<pad>'] = 0
    vocab['<sos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    print(f"Vocabulary size: {len(vocab)}")
    return vocab

# Transformer-based Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, n_heads, hid_dim, n_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(emb_dim, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hid_dim, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask):
        # src: [seq_len, batch_size]
        embedded = self.embedding(src) * np.sqrt(self.embedding.embedding_dim)  # [seq_len, batch_size, emb_dim]
        embedded = self.positional_encoding(embedded)
        output = self.transformer_encoder(embedded, src_key_padding_mask=src_key_padding_mask)  # [seq_len, batch_size, emb_dim]
        return output

# Transformer-based Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, n_heads, hid_dim, n_layers, dropout):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(emb_dim, dropout=dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model=emb_dim, nhead=n_heads, dim_feedforward=hid_dim, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=n_layers)
        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, memory, trg_mask, memory_key_padding_mask):
        # trg: [trg_seq_len, batch_size]
        embedded = self.embedding(trg) * np.sqrt(self.embedding.embedding_dim)  # [trg_seq_len, batch_size, emb_dim]
        embedded = self.positional_encoding(embedded)
        output = self.transformer_decoder(tgt=embedded,
                                          memory=memory,
                                          tgt_mask=trg_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)  # [trg_seq_len, batch_size, emb_dim]
        output = self.fc_out(output)  # [trg_seq_len, batch_size, output_dim]
        return output

# Complete Transformer-based Seq2Seq Model
class TransformerSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super(TransformerSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx

    def make_src_key_padding_mask(self, src):
        # src: [seq_len, batch_size]
        # mask: [batch_size, seq_len] -> True for padding positions
        return (src == self.pad_idx).transpose(0, 1)

    def make_trg_key_padding_mask(self, trg):
        # trg: [trg_seq_len, batch_size]
        # mask: [batch_size, trg_seq_len] -> True for padding positions
        return (trg == self.pad_idx).transpose(0, 1)

    def make_trg_mask(self, trg_seq_len):
        # trg_mask: [trg_seq_len, trg_seq_len]
        trg_mask = nn.Transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        return trg_mask

    def forward(self, src, trg):
        # src: [batch_size, src_seq_len] -> [src_seq_len, batch_size]
        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)
        src_key_padding_mask = self.make_src_key_padding_mask(src)
        trg_key_padding_mask = self.make_trg_key_padding_mask(trg)
        trg_mask = self.make_trg_mask(trg.size(0))

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)  # [src_seq_len, batch_size, emb_dim]
        output = self.decoder(trg, memory, trg_mask=trg_mask, memory_key_padding_mask=src_key_padding_mask)  # [trg_seq_len, batch_size, output_dim]
        return output

# Data Loading Function
def load_data(filename, start_index=None, end_index=None):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    if start_index is not None and end_index is not None:
        data = data.iloc[start_index:end_index]
        print(f"Data sliced from index {start_index} to {end_index}")
    print(f"Total samples loaded: {len(data)}")
    return data

# Tokenization Function
def tokenize(text):
    return list(text.lower())

# Training Function
def train(model, iterator, optimizer, criterion, epoch, args):
    model.train()
    epoch_loss = 0
    print(f"Starting training for epoch {epoch+1}...")
    for i, (src, trg) in enumerate(iterator):
        src = src.to(model.device)  # [batch_size, src_seq_len]
        trg = trg.to(model.device)  # [batch_size, trg_seq_len]

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # Exclude the last token for input to the decoder

        # output: [trg_seq_len -1, batch_size, output_dim]
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # Exclude the first token (<sos>)

        loss = criterion(output, trg)
        loss.backward()

        if args.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)

        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Batch [{i+1}/{len(iterator)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(iterator)
    print(f"Epoch [{epoch+1}] Training Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

# Evaluation Function
def evaluate(model, iterator, criterion, target_vocab, epoch):
    model.eval()
    epoch_loss = 0
    wers = []
    inv_target_vocab = {idx: ch for ch, idx in target_vocab.items()}
    print(f"Starting evaluation for epoch {epoch+1}...")

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(model.device)
            trg = trg.to(model.device)

            output = model(src, trg[:, :-1])

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            # Calculate WER
            output_tokens = output.argmax(1).view(trg.size(0) // trg.size(1), trg.size(1))
            trg_tokens = trg.view(trg.size(0) // trg.size(1), trg.size(1))

            for pred_seq, trg_seq in zip(output_tokens, trg_tokens):
                pred_seq = pred_seq.cpu().numpy()
                trg_seq = trg_seq.cpu().numpy()

                # Convert indices to characters
                pred_chars = [inv_target_vocab.get(idx, '') for idx in pred_seq]
                trg_chars = [inv_target_vocab.get(idx, '') for idx in trg_seq]

                # Remove special tokens
                pred_text = ''.join([ch for ch in pred_chars if ch not in ['<pad>', '<sos>', '<eos>']])
                trg_text = ''.join([ch for ch in trg_chars if ch not in ['<pad>', '<sos>', '<eos>']])

                # Calculate WER
                wer_score = wer(trg_text, pred_text)
                wers.append(wer_score)

            if (i + 1) % 50 == 0:
                print(f"Evaluation Batch [{i+1}/{len(iterator)}], Current Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(iterator)
    avg_wer = sum(wers) / len(wers) if len(wers) > 0 else 0
    print(f"Epoch [{epoch+1}] Evaluation Completed. Average Loss: {avg_loss:.4f}, Average WER: {avg_wer:.4f}")
    return avg_loss, avg_wer

# Inference Function
def translate_sentence(model, sentence, input_vocab, target_vocab, max_length=100):
    model.eval()
    tokens = [input_vocab.get(ch, input_vocab['<unk>']) for ch in list(sentence.lower())]
    src = torch.tensor([input_vocab['<sos>']] + tokens + [input_vocab['<eos>']]).unsqueeze(0).to(model.device)  # [1, src_seq_len]
    src_key_padding_mask = (src == input_vocab['<pad>']).transpose(0, 1)  # [1, src_seq_len]

    memory = model.encoder(src.transpose(0,1), src_key_padding_mask=src_key_padding_mask)  # [src_seq_len, 1, emb_dim]

    trg_indices = [target_vocab['<sos>']]

    for _ in range(max_length):
        trg = torch.tensor(trg_indices).unsqueeze(1).to(model.device)  # [trg_seq_len, 1]
        trg_mask = model.make_trg_mask(trg.size(0))

        output = model.decoder(trg, memory, trg_mask=trg_mask, memory_key_padding_mask=src_key_padding_mask)  # [trg_seq_len, 1, output_dim]
        pred_token = output[-1, 0, :].argmax(dim=-1).item()

        if pred_token == target_vocab['<eos>']:
            break
        trg_indices.append(pred_token)

    inv_target_vocab = {idx: ch for ch, idx in target_vocab.items()}
    translated_sentence = ''.join([inv_target_vocab.get(idx, '') for idx in trg_indices[1:]])  # Exclude <sos>
    return translated_sentence

# Main Function
def main(args):
    print("Initializing training process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    train_data = load_data(args.train_filename, args.start_index, args.end_index)
    val_data = load_data(args.validation_filename)
    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()

    print("Building vocabularies...")
    # Build vocabularies
    input_vocab = build_char_vocab(train_input_texts + val_input_texts)
    target_vocab = build_char_vocab(train_target_texts + val_target_texts)

    print("Creating datasets and dataloaders...")
    # Create datasets and dataloaders
    train_dataset = TextCorrectionDataset(train_input_texts, train_target_texts, input_vocab, target_vocab, args.max_length)
    val_dataset = TextCorrectionDataset(val_input_texts, val_target_texts, input_vocab, target_vocab, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    input_dim = len(input_vocab)
    output_dim = len(target_vocab)
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

    # Instantiate the Transformer components
    encoder = TransformerEncoder(input_dim=input_dim,
                                 emb_dim=args.emb_dim,
                                 n_heads=args.n_heads,
                                 hid_dim=args.hid_dim,
                                 n_layers=args.n_layers,
                                 dropout=args.dropout_rate)

    decoder = TransformerDecoder(output_dim=output_dim,
                                 emb_dim=args.emb_dim,
                                 n_heads=args.n_heads,
                                 hid_dim=args.hid_dim,
                                 n_layers=args.n_layers,
                                 dropout=args.dropout_rate)

    # Create the Seq2Seq Transformer model
    model = TransformerSeq2Seq(encoder, decoder, device, pad_idx=input_vocab['<pad>']).to(device)
    print("Model initialized.")

    # Adjust optimizer with or without weight decay
    if args.use_weight_decay:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        print(f"Using weight decay with coefficient: {args.weight_decay}")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        print("Weight decay not used.")

    # Using label smoothing in the loss function
    if args.use_label_smoothing:
        criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing_value)
        print(f"Using label smoothing with value: {args.label_smoothing_value}")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        print("Label smoothing not used.")

    # Training loop with optional early stopping
    best_val_wer = float('inf')
    patience_counter = 0
    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
        train_loss = train(model, train_loader, optimizer, criterion, epoch, args)
        val_loss, val_wer = evaluate(model, val_loader, criterion, target_vocab, epoch=epoch)
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")

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

    # Save the final model and vocabularies
    print("Training completed. Saving final model and vocabularies...")
    torch.save(model.state_dict(), 'transformer_seq2seq_model.pth')
    with open('input_vocab.pkl', 'wb') as f:
        pickle.dump(input_vocab, f)
    with open('target_vocab.pkl', 'wb') as f:
        pickle.dump(target_vocab, f)
    print("Model and vocabularies saved.")

    # Evaluation on the validation set
    print("\n=== Final Evaluation on Validation Set ===")
    wers = []
    corrected_texts = []
    for idx, (input_text, target_text) in enumerate(tqdm(zip(val_input_texts, val_target_texts), total=len(val_input_texts))):
        prediction = translate_sentence(model, input_text, input_vocab, target_vocab, args.max_length)
        corrected_texts.append(prediction)
        wer_score = wer(target_text, prediction)
        wers.append(wer_score)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{len(val_input_texts)} samples. Current WER: {wer_score:.4f}")

    average_wer = sum(wers) / len(wers)
    print(f'Final Validation Average WER: {average_wer:.4f}')

    # Save the validation results
    print("Saving validation results...")
    val_results = pd.DataFrame({
        'corrupt_msg': val_input_texts,
        'gold_msg': val_target_texts,
        'corrected_msg': corrected_texts,
        'wer': wers
    })
    val_results.to_csv('validation_results.csv', index=False)
    print("Validation results saved to 'validation_results.csv'.")

# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer-Based Sequence-to-Sequence Autocorrect Training Script with Configurable Options")
    parser.add_argument('--train_filename', type=str, default='train_fold.csv', help='Path to training data CSV file')
    parser.add_argument('--validation_filename', type=str, default='val_fold.csv', help='Path to validation data CSV file')
    parser.add_argument('--start_index', type=int, default=None, help='Start index for training data slicing')
    parser.add_argument('--end_index', type=int, default=None, help='End index for training data slicing')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension size')
    parser.add_argument('--hid_dim', type=int, default=512, help='Hidden dimension size for Transformer feedforward layers')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of layers in the Transformer')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads in the Transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length for padding')
    parser.add_argument('--use_weight_decay', action='store_true', help='Use weight decay (L2 regularization)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay coefficient')
    parser.add_argument('--use_label_smoothing', action='store_true', help='Use label smoothing in the loss function')
    parser.add_argument('--label_smoothing_value', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--use_early_stopping', action='store_true', help='Use early stopping during training')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--use_gradient_clipping', action='store_true', help='Use gradient clipping during training')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Maximum norm for gradient clipping')
    # Removed teacher forcing ratio as it's not directly applicable to Transformer
    args = parser.parse_args()

    main(args)
