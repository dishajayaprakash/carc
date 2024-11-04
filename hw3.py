#!/usr/bin/env python

import argparse
import string
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# ------------------------------
# Device Configuration
# ------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------
# Model Definition (Must Match Training)
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
# Helper Functions
# ------------------------------

def load_vocab(vocab_path='vocab.npz'):
    """
    Load the vocabulary mappings from a .npz file.

    Parameters:
        vocab_path (str): Path to the vocab .npz file.

    Returns:
        tuple: (input_vocab, idx_to_char)
    """
    try:
        data = np.load(vocab_path, allow_pickle=True)
        input_vocab = data['input_vocab'].item()
        idx_to_char = data['idx_to_char'].item()
        print(f"Loaded vocabulary from {vocab_path}.")
        return input_vocab, idx_to_char
    except FileNotFoundError:
        print(f"Error: Vocabulary file {vocab_path} not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading vocabulary: {e}")
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

def sequence_to_text(seq, idx_to_char):
    """
    Convert a sequence of indices back to text.

    Parameters:
        seq (list of int): Sequence of indices.
        idx_to_char (dict): Index to character mapping.

    Returns:
        str: Decoded text.
    """
    return ''.join([idx_to_char.get(idx, '') for idx in seq]).strip()

def load_model(model_path, vocab_size, embedding_dim, lstm_units, dropout_rate):
    """
    Initialize and load the model's state dictionary.

    Parameters:
        model_path (str): Path to the saved model .pth file.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the embedding layer.
        lstm_units (int): Number of LSTM units.
        dropout_rate (float): Dropout rate.

    Returns:
        nn.Module: Loaded model.
    """
    model = AutocorrectModel(vocab_size, embedding_dim, lstm_units, dropout_rate).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded model state from {model_path}.")
        return model
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found.")
        exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit(1)

# ------------------------------
# Prediction Function
# ------------------------------

def predict(model, input_text, input_vocab, idx_to_char, max_length):
    """
    Perform autocorrect prediction on a single input text.

    Parameters:
        model (nn.Module): Trained autocorrect model.
        input_text (str): The corrupted input text.
        input_vocab (dict): Character to index mapping.
        idx_to_char (dict): Index to character mapping.
        max_length (int): Maximum sequence length.

    Returns:
        str: Corrected text.
    """
    preprocessed_text = preprocess_text(input_text)
    seq = text_to_sequence(preprocessed_text, max_length, input_vocab)
    input_tensor = torch.tensor([seq], dtype=torch.long).to(device)  # Batch size = 1
    with torch.no_grad():
        output = model(input_tensor)  # (1, seq_length, vocab_size)
        predicted_indices = output.argmax(dim=-1).cpu().numpy()[0]  # (seq_length,)
    corrected_text = sequence_to_text(predicted_indices, idx_to_char)
    return corrected_text

# ------------------------------
# Batch Prediction Function
# ------------------------------

def batch_predict(model, input_texts, input_vocab, idx_to_char, max_length, batch_size=32):
    """
    Perform autocorrect prediction on a batch of input texts.

    Parameters:
        model (nn.Module): Trained autocorrect model.
        input_texts (list of str): List of corrupted input texts.
        input_vocab (dict): Character to index mapping.
        idx_to_char (dict): Index to character mapping.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for prediction.

    Returns:
        list of str: List of corrected texts.
    """
    corrected_texts = []
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i+batch_size]
        preprocessed_texts = [preprocess_text(text) for text in batch_texts]
        sequences = [text_to_sequence(text, max_length, input_vocab) for text in preprocessed_texts]
        input_tensor = torch.tensor(sequences, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)  # (batch_size, seq_length, vocab_size)
            predicted_indices = outputs.argmax(dim=-1).cpu().numpy()  # (batch_size, seq_length)
        for seq in predicted_indices:
            corrected_text = sequence_to_text(seq, idx_to_char)
            corrected_texts.append(corrected_text)
    return corrected_texts

# ------------------------------
# Main Prediction Script
# ------------------------------

def main(args):
    # Load vocabulary
    input_vocab, idx_to_char = load_vocab(args.vocab_path)
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        vocab_size=len(input_vocab),
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    )
    
    # Determine max_length
    if args.max_seq_length is not None:
        max_length = args.max_seq_length
        print(f"Using provided max sequence length: {max_length}")
    else:
        # If not provided, determine from the training data
        # This assumes you have the max_seq_length from training saved or known
        # For simplicity, we'll set a default value
        max_length = 100  # Change this based on your training configuration
        print(f"No max sequence length provided. Using default: {max_length}")
    
    # Handle different modes: single input or batch input
    if args.input_text:
        # Single input prediction
        corrected = predict(model, args.input_text, input_vocab, idx_to_char, max_length)
        print(f"\nCorrupted Message: {args.input_text}")
        print(f"Corrected Message: {corrected}")
    
    elif args.input_csv:
        # Batch prediction from a CSV file with 'corrupt_msg' column
        try:
            df_test = pd.read_csv(args.input_csv)
            if 'corrupt_msg' not in df_test.columns:
                print(f"Error: 'corrupt_msg' column not found in {args.input_csv}.")
                exit(1)
            input_texts = df_test['corrupt_msg'].astype(str).tolist()
            print(f"\nLoaded {len(input_texts)} messages from {args.input_csv} for prediction.")
        except FileNotFoundError:
            print(f"Error: Input file {args.input_csv} not found.")
            exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: Input file {args.input_csv} is empty.")
            exit(1)
        except Exception as e:
            print(f"An error occurred while reading {args.input_csv}: {e}")
            exit(1)
        
        # Perform batch prediction
        corrected_texts = batch_predict(model, input_texts, input_vocab, idx_to_char, max_length, batch_size=args.batch_size)
        
        # Save predictions to a CSV file
        df_output = pd.DataFrame({
            'corrupt_msg': input_texts,
            'pred_msg': corrected_texts
        })
        df_output.to_csv(args.output_csv, index=False)
        print(f"Predictions saved to {args.output_csv}.")
    
    else:
        print("Error: No input provided for prediction. Use --input_text for single prediction or --input_csv for batch prediction.")
        exit(1)

# ------------------------------
# Argument Parsing
# ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autocorrect Prediction Script")
    
    # Mutually exclusive group for single or batch prediction
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_text', type=str, help='Single corrupted message to autocorrect.')
    group.add_argument('--input_csv', type=str, help='Path to a CSV file containing corrupted messages with a "corrupt_msg" column.')
    
    # Output arguments
    parser.add_argument('--output_csv', type=str, default='test_fold_predictions.csv', help='Output CSV file to save predictions (used with --input_csv).')
    
    # Model and vocabulary paths
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the saved model .pth file.')
    parser.add_argument('--vocab_path', type=str, default='vocab.npz', help='Path to the vocabulary .npz file.')
    
    # Sequence length
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length for padding (must match training).')
    
    # Batch size for predictions
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for batch prediction.')
    
    # Model hyperparameters (must match training)
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of embedding layer.')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate.')
    
    args = parser.parse_args()
    main(args)
