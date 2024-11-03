#!/usr/bin/env python

import os
import argparse
import pandas as pd
import string
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import jiwer  # For WER calculation

# ------------------------------
# Environment and Threading Setup
# ------------------------------

# Limit the number of threads used by OpenBLAS and other libraries to prevent resource exhaustion
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Set the number of threads used by PyTorch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ------------------------------
# Device Configuration
# ------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# Data Loading and Preprocessing
# ------------------------------

def load_data(filename, max_samples=None):
    """
    Load the dataset from a CSV file.

    Parameters:
        filename (str): Path to the CSV file.
        max_samples (int, optional): Maximum number of samples to load.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        df = pd.read_csv(filename, nrows=max_samples)
        print(f"Loaded {len(df)} samples from {filename}.")
        return df
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {filename} is empty.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading {filename}: {e}")
        exit(1)

def preprocess_text(text):
    """
    Lowercase and remove punctuation from text.

    Parameters:
        text (str): Input text.

    Returns:
        str: Preprocessed text.
    """
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def build_char_vocab(texts):
    """
    Build character to index and index to character mappings.

    Parameters:
        texts (list of str): List of texts.

    Returns:
        tuple: (char_to_idx, idx_to_char)
    """
    all_chars = set(''.join(texts))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(all_chars))}
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = len(char_to_idx)
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    return char_to_idx, idx_to_char

def text_to_sequence(text, max_length, char_to_idx):
    """
    Convert text to a sequence of indices.

    Parameters:
        text (str): Input text.
        max_length (int): Maximum sequence length.
        char_to_idx (dict): Character to index mapping.

    Returns:
        list of int: Sequence of indices.
    """
    seq = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]
    seq = seq[:max_length]
    seq += [char_to_idx['<PAD>']] * (max_length - len(seq))
    return seq

def validate_data(input_texts, target_texts, dataset_type="Training"):
    """
    Ensure that no target text is empty.

    Parameters:
        input_texts (list of str): List of input texts.
        target_texts (list of str): List of target texts.
        dataset_type (str): Type of dataset (e.g., "Training", "Validation").

    Returns:
        tuple: Filtered (input_texts, target_texts)
    """
    filtered = [
        (inp, tgt) for inp, tgt in zip(input_texts, target_texts)
        if inp.strip() != '' and tgt.strip() != ''
    ]
    num_removed = len(input_texts) - len(filtered)
    if num_removed > 0:
        print(f"{dataset_type} data: Removed {num_removed} samples with empty 'corrupt_msg' or 'gold_msg'.")
    else:
        print(f"{dataset_type} data: No empty 'corrupt_msg' or 'gold_msg' entries found.")
    filtered_input, filtered_target = zip(*filtered) if filtered else ([], [])
    return list(filtered_input), list(filtered_target)

# ------------------------------
# Dataset Definition
# ------------------------------

class TextDataset(Dataset):
    """
    Custom Dataset for text data.
    """
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

# ------------------------------
# Model Definition
# ------------------------------

class AutocorrectModel(nn.Module):
    """
    Character-Level Autocorrect Model using Bidirectional LSTM.
    """
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

# ------------------------------
# Training and Evaluation
# ------------------------------

def main(args):
    print("Initializing training process...")

    # Load and preprocess training data
    print("\nLoading training data...")
    train_data = load_data(args.train_filename, args.max_samples_train)
    train_data['corrupt_msg'] = train_data['corrupt_msg'].astype(str).apply(preprocess_text)
    train_data['gold_msg'] = train_data['gold_msg'].astype(str).apply(preprocess_text)
    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    train_input_texts, train_target_texts = validate_data(train_input_texts, train_target_texts, dataset_type="Training")

    # Load and preprocess validation data
    print("\nLoading validation data...")
    val_data = load_data(args.validation_filename, args.max_samples_val)
    val_data['corrupt_msg'] = val_data['corrupt_msg'].astype(str).apply(preprocess_text)
    val_data['gold_msg'] = val_data['gold_msg'].astype(str).apply(preprocess_text)
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()
    val_input_texts, val_target_texts = validate_data(val_input_texts, val_target_texts, dataset_type="Validation")

    print(f"\nTraining samples after filtering: {len(train_input_texts)}")
    print(f"Validation samples after filtering: {len(val_input_texts)}")

    # Build vocabularies
    print("\nBuilding vocabularies...")
    input_vocab, idx_to_char = build_char_vocab(train_input_texts + train_target_texts + val_input_texts + val_target_texts)
    vocab_size = len(input_vocab)
    print(f"Vocabulary size (including <PAD> and <UNK>): {vocab_size}")

    # Determine maximum sequence length
    max_seq_length_train = max(len(text) for text in train_input_texts + train_target_texts)
    max_seq_length_val = max(len(text) for text in val_input_texts + val_target_texts)
    max_seq_length = max(max_seq_length_train, max_seq_length_val)
    print(f"Determined max sequence length: {max_seq_length}")

    # Adjust max_seq_length if provided
    if args.max_seq_length is not None:
        max_seq_length = args.max_seq_length
        print(f"Using provided max sequence length: {max_seq_length}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_input_texts, train_target_texts, max_seq_length, input_vocab)
    val_dataset = TextDataset(val_input_texts, val_target_texts, max_seq_length, input_vocab)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    # Build model
    print("\nBuilding the model...")
    model = AutocorrectModel(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    ).to(device)

    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction='none')  # Use negative log likelihood loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets, sample_weights) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # (batch_size, seq_length, vocab_size)
            outputs = outputs.permute(0, 2, 1)  # (batch_size, vocab_size, seq_length)

            loss = criterion(outputs, targets)  # (batch_size, seq_length)
            loss = loss * sample_weights  # Apply sample weights
            loss = loss.sum() / sample_weights.sum()  # Average loss over non-padded tokens

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation loop
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
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Validation Loss: {avg_val_loss:.4f}")

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Early Stopping Logic
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model.")
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for epoch {epoch+1}. Patience counter: {patience_counter}")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

        # WER Calculation on Validation Set
        print("\nCalculating WER on validation set...")
        model.eval()
        predicted_texts = []
        reference_texts = []
        with torch.no_grad():
            for inputs, targets, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)  # (batch_size, seq_length, vocab_size)
                predicted_indices = outputs.argmax(dim=-1).cpu().numpy()  # (batch_size, seq_length)
                inputs = inputs.cpu().numpy()

                for i in range(inputs.shape[0]):
                    input_seq = inputs[i]
                    gold_seq = targets[i][:max_seq_length].cpu().numpy()
                    gold_length = np.count_nonzero(gold_seq)

                    if gold_length == 0:
                        # This should not happen due to prior filtering, but added for safety
                        print(f"Sample {i} has gold_length=0. Skipping.")
                        continue

                    pred_seq = predicted_indices[i][:gold_length]
                    pred_text = ''.join([idx_to_char.get(idx, '') for idx in pred_seq]).strip()
                    gold_text = ''.join([idx_to_char.get(idx, '') for idx in gold_seq[:gold_length]]).strip()

                    # Additional checks
                    if gold_text == '':
                        print(f"Sample {i}: 'gold_text' is empty despite prior filtering. Skipping.")
                        continue
                    if pred_text == '':
                        print(f"Sample {i}: 'pred_text' is empty. Skipping.")
                        continue

                    predicted_texts.append(pred_text)
                    reference_texts.append(gold_text)

        # Compute WER
        if reference_texts and predicted_texts:
            wer_scores = [jiwer.wer(ref, hyp) for ref, hyp in zip(reference_texts, predicted_texts)]
            average_wer = np.mean(wer_scores) * 100
            print(f"Epoch [{epoch+1}/{args.num_epochs}] - Validation WER: {average_wer:.2f}%\n")
        else:
            print("No valid samples for WER calculation.\n")

    # Load the best model for final evaluation
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
        print("Loaded the best model based on validation loss.")
    else:
        print("Best model not found. Proceeding with the current model.")

    # Final Evaluation on the Validation Set
    print("\n=== Final Evaluation on Validation Set ===")
    model.eval()
    predicted_texts = []
    wers = []
    with torch.no_grad():
        val_loader_final = DataLoader(val_dataset, batch_size=args.batch_size)
        for inputs, targets, _ in val_loader_final:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predicted_indices = outputs.argmax(dim=-1).cpu().numpy()  # (batch_size, seq_length)
            inputs = inputs.cpu().numpy()

            for i in range(inputs.shape[0]):
                input_seq = inputs[i]
                gold_seq = targets[i][:max_seq_length].cpu().numpy()
                gold_length = np.count_nonzero(gold_seq)

                if gold_length == 0:
                    # This should not happen due to prior filtering
                    print(f"Sample {i} has gold_length=0. Skipping.")
                    continue

                pred_seq = predicted_indices[i][:gold_length]
                pred_text = ''.join([idx_to_char.get(idx, '') for idx in pred_seq]).strip()
                gold_text = ''.join([idx_to_char.get(idx, '') for idx in gold_seq[:gold_length]]).strip()

                # Additional checks
                if gold_text == '':
                    print(f"Sample {i}: 'gold_text' is empty despite prior filtering. Skipping.")
                    continue
                if pred_text == '':
                    print(f"Sample {i}: 'pred_text' is empty. Skipping.")
                    continue

                predicted_texts.append(pred_text)
                gold_text = gold_text
                wers.append(jiwer.wer(gold_text, pred_text))

                if len(predicted_texts) % 1000 == 0:
                    print(f"Processed {len(predicted_texts)}/{len(val_input_texts)} samples. Current WER: {wers[-1]:.4f}")

    if wers:
        average_wer = np.mean(wers) * 100
        print(f'Final Validation Average WER: {average_wer:.2f}%')
    else:
        print("No valid samples for final WER calculation.")

    # Save validation results
    print("\nSaving validation results...")
    val_results = pd.DataFrame({
        'corrupt_msg': val_input_texts[:len(wers)],
        'gold_msg': val_target_texts[:len(wers)],
        'corrected_msg': predicted_texts,
        'wer': wers
    })
    val_results.to_csv('validation_results.csv', index=False)
    print("Validation results saved to 'validation_results.csv'.")

    # Save the vocabulary
    print("\nSaving vocabulary...")
    vocab_path = 'vocab.npz'
    np.savez(vocab_path, input_vocab=input_vocab, idx_to_char=idx_to_char)
    print("Vocabulary saved.")

# ------------------------------
# Argument Parsing
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character-Level Autocorrect Model Training Script with PyTorch")
    parser.add_argument('--train_filename', type=str, default='train_fold.csv', help='Path to training data CSV file')
    parser.add_argument('--validation_filename', type=str, default='val_fold.csv', help='Path to validation data CSV file')
    parser.add_argument('--max_samples_train', type=int, default=None, help='Maximum number of training samples to read from the dataset')
    parser.add_argument('--max_samples_val', type=int, default=None, help='Maximum number of validation samples to read from the dataset')
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
