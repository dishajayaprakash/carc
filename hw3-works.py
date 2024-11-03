#!/usr/bin/env python

import argparse
import pandas as pd
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import jiwer  # For WER calculation
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_data(filename, max_samples=None):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filename, nrows=max_samples)
    return df

def preprocess_text(text):
    """Lowercase and remove punctuation from text."""
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def build_char_vocab(texts):
    """Build character to index and index to character mappings."""
    all_chars = set(''.join(texts))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def text_to_sequence(text, max_length, char_to_idx):
    """Convert text to a sequence of indices."""
    seq = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]
    seq = seq[:max_length]
    seq += [char_to_idx['<PAD>']] * (max_length - len(seq))
    return seq

class TextDataset(Dataset):
    def __init__(self, input_texts, target_texts, max_length, char_to_idx):
        self.inputs = [text_to_sequence(text, max_length, char_to_idx) for text in input_texts]
        self.targets = [text_to_sequence(text, max_length, char_to_idx) for text in target_texts]
        self.sample_weights = [
            [0.0 if idx == char_to_idx['<PAD>'] else 1.0 for idx in seq] for seq in self.targets
        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.long)
        target_seq = torch.tensor(self.targets[idx], dtype=torch.long)
        sample_weight = torch.tensor(self.sample_weights[idx], dtype=torch.float)
        return input_seq, target_seq, sample_weight

class AutocorrectModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_units, dropout_rate):
        super(AutocorrectModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(
            embedding_dim,
            lstm_units,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_units * 2, vocab_size)  # *2 for bidirectional
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        x, _ = self.lstm(x)
        x = self.lstm_dropout(x)
        x = self.fc(x)
        x = self.log_softmax(x)
        return x

def main(args):
    print("Initializing training process...")

    # Load and preprocess data
    print("Loading data...")
    train_data = load_data(args.train_filename, args.max_samples)
    val_data = load_data(args.validation_filename, args.max_samples)

    # Preprocess the data: lowercase and remove punctuation
    train_data['corrupt_msg'] = train_data['corrupt_msg'].astype(str).apply(preprocess_text)
    train_data['gold_msg'] = train_data['gold_msg'].astype(str).apply(preprocess_text)

    val_data['corrupt_msg'] = val_data['corrupt_msg'].astype(str).apply(preprocess_text)
    val_data['gold_msg'] = val_data['gold_msg'].astype(str).apply(preprocess_text)

    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()

    print(f"Training samples: {len(train_input_texts)}")
    print(f"Validation samples: {len(val_input_texts)}")

    # Build vocabularies
    print("Building vocabularies...")
    input_vocab, idx_to_char = build_char_vocab(train_input_texts + train_target_texts + val_input_texts + val_target_texts)
    vocab_size = len(input_vocab)
    print(f"Vocabulary size (including <PAD> and <UNK>): {vocab_size}")

    # Determine maximum sequence length
    max_seq_length = max(
        max(len(text) for text in train_input_texts + train_target_texts),
        max(len(text) for text in val_input_texts + val_target_texts)
    )
    print(f"Max sequence length: {max_seq_length}")

    # Adjust max_seq_length if provided
    if args.max_seq_length is not None:
        max_seq_length = args.max_seq_length
        print(f"Using provided max sequence length: {max_seq_length}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = TextDataset(train_input_texts, train_target_texts, max_seq_length, input_vocab)
    val_dataset = TextDataset(val_input_texts, val_target_texts, max_seq_length, input_vocab)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Build model
    print("Building model...")
    model = AutocorrectModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction='none')  # Use negative log likelihood loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets, sample_weights in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)  # (batch_size, vocab_size, seq_length)

            loss = criterion(outputs, targets)
            loss = loss * sample_weights  # Apply sample weights
            loss = loss.sum() / sample_weights.sum()  # Average loss over non-padded tokens

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, targets, sample_weights in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)

                outputs = model(inputs)
                outputs = outputs.permute(0, 2, 1)

                loss = criterion(outputs, targets)
                loss = loss * sample_weights
                loss = loss.sum() / sample_weights.sum()
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model improved. Saving model.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        # WER Calculation on Validation Set
        print("Calculating WER on validation set...")
        model.eval()
        predicted_texts = []
        reference_texts = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predicted_indices = outputs.argmax(dim=-1).cpu().numpy()
                inputs = inputs.cpu().numpy()

                for i in range(inputs.shape[0]):
                    input_seq = inputs[i]
                    input_length = np.count_nonzero(input_seq)
                    pred_seq = predicted_indices[i][:input_length]
                    pred_text = ''.join([idx_to_char.get(idx, '') for idx in pred_seq])
                    pred_text = pred_text.strip()
                    predicted_texts.append(pred_text)

                    gold_seq = targets[i][:input_length].cpu().numpy()
                    gold_text = ''.join([idx_to_char.get(idx, '') for idx in gold_seq])
                    gold_text = gold_text.strip()
                    reference_texts.append(gold_text)

        wer_scores = [jiwer.wer(ref, hyp) for ref, hyp in zip(reference_texts, predicted_texts)]
        average_wer = np.mean(wer_scores) * 100
        print(f"Epoch {epoch+1} - Validation WER: {average_wer:.2f}%")

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluation on the validation set
    print("\n=== Final Evaluation on Validation Set ===")
    model.eval()
    predicted_texts = []
    wers = []
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        for inputs, targets, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_indices = outputs.argmax(dim=-1).cpu().numpy()
            inputs = inputs.cpu().numpy()

            for i in range(inputs.shape[0]):
                input_seq = inputs[i]
                input_length = np.count_nonzero(input_seq)
                pred_seq = predicted_indices[i][:input_length]
                pred_text = ''.join([idx_to_char.get(idx, '') for idx in pred_seq])
                pred_text = pred_text.strip()
                predicted_texts.append(pred_text)

                gold_text = val_target_texts[len(predicted_texts)-1]
                wer_score = jiwer.wer(gold_text, pred_text)
                wers.append(wer_score)

                if len(predicted_texts) % 1000 == 0:
                    print(f"Processed {len(predicted_texts)}/{len(val_input_texts)} samples. Current WER: {wer_score:.4f}")

    average_wer = np.mean(wers) * 100
    print(f'Final Validation Average WER: {average_wer:.2f}%')

    # Save validation results
    print("Saving validation results...")
    val_results = pd.DataFrame({
        'corrupt_msg': val_input_texts,
        'gold_msg': val_target_texts,
        'corrected_msg': predicted_texts,
        'wer': wers
    })
    val_results.to_csv('validation_results.csv', index=False)
    print("Validation results saved to 'validation_results.csv'.")

    # Save the vocabulary
    print("Saving vocabulary...")
    vocab_path = 'vocab.npz'
    np.savez(vocab_path, input_vocab=input_vocab, idx_to_char=idx_to_char)
    print("Vocabulary saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character-Level Autocorrect Model Training Script with PyTorch")
    parser.add_argument('--train_filename', type=str, default='train_fold.csv', help='Path to training data CSV file')
    parser.add_argument('--validation_filename', type=str, default='val_fold.csv', help='Path to validation data CSV file')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to read from the dataset')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and evaluation')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of embedding layer')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length for padding')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()
    main(args)
