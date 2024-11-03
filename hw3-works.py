#!/usr/bin/env python

import argparse
import pandas as pd
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
import jiwer  # For WER calculation
import os

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
    return pad_sequences([seq], maxlen=max_length, padding='post', value=char_to_idx['<PAD>'])[0]

def create_dataset(input_texts, target_texts, max_length, char_to_idx):
    """Create input and target sequences and sample weights."""
    inputs = np.array([text_to_sequence(text, max_length, char_to_idx) for text in input_texts])
    targets = np.array([text_to_sequence(text, max_length, char_to_idx) for text in target_texts])
    # Expand dimensions for targets to match required shape
    targets = np.expand_dims(targets, -1)
    # Create sample weights to mask the loss over padded positions
    sample_weights = np.where(targets.squeeze(-1) == char_to_idx['<PAD>'], 0.0, 1.0)
    return inputs, targets, sample_weights

def build_model(vocab_size, max_seq_length, embedding_dim, lstm_units, dropout_rate):
    """Build the encoder-only model."""
    inputs = Input(shape=(max_seq_length,), name='input')
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='embedding'
    )(inputs)
    embedding = Dropout(dropout_rate, name='embedding_dropout')(embedding)
    encoder_outputs = Bidirectional(
        LSTM(lstm_units, return_sequences=True),
        name='bidirectional_lstm'
    )(embedding)
    encoder_outputs = Dropout(dropout_rate, name='lstm_dropout')(encoder_outputs)
    outputs = TimeDistributed(
        Dense(vocab_size, activation='softmax'),
        name='time_distributed_dense'
    )(encoder_outputs)
    model = Model(inputs, outputs)
    return model

def main(args):
    print("Initializing training process...")

    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available.")
        # Optional: Set memory growth to prevent TensorFlow from allocating all GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Error setting memory growth: {e}")
    else:
        print("GPU is not available. Using CPU.")

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
    X_train, y_train, sample_weights_train = create_dataset(train_input_texts, train_target_texts, max_seq_length, input_vocab)
    X_val, y_val, sample_weights_val = create_dataset(val_input_texts, val_target_texts, max_seq_length, input_vocab)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # Build model
    print("Building model...")
    model = build_model(
        vocab_size=vocab_size,
        max_seq_length=max_seq_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate
    )

    # Compile the model with weighted_metrics to suppress the warning
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    model.summary()

    # Implement callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # WER Callback
    class WERCallback(Callback):
        def __init__(self, validation_data, idx_to_char):
            super(WERCallback, self).__init__()
            self.validation_data = validation_data
            self.idx_to_char = idx_to_char

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val = self.validation_data
            predictions = self.model.predict(X_val)
            predicted_indices = np.argmax(predictions, axis=-1)

            predicted_texts = []
            reference_texts = []

            for i in range(len(X_val)):
                predicted_seq = predicted_indices[i]
                input_seq = X_val[i]
                input_length = np.count_nonzero(input_seq)
                predicted_seq = predicted_seq[:input_length]
                predicted_text = ''.join([self.idx_to_char.get(idx, '') for idx in predicted_seq])
                predicted_text = predicted_text.strip()
                predicted_texts.append(predicted_text)

                gold_seq = y_val[i].squeeze()
                gold_seq = gold_seq[:input_length]
                gold_text = ''.join([self.idx_to_char.get(idx, '') for idx in gold_seq])
                gold_text = gold_text.strip()
                reference_texts.append(gold_text)

            wer_scores = [jiwer.wer(ref, hyp) for ref, hyp in zip(reference_texts, predicted_texts)]
            average_wer = np.mean(wer_scores) * 100
            print(f"Epoch {epoch+1} - Validation WER: {average_wer:.2f}%")

    wer_callback = WERCallback(validation_data=(X_val, y_val), idx_to_char=idx_to_char)

    # Training loop
    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        validation_data=(X_val, y_val, sample_weights_val),
        callbacks=[early_stopping, reduce_lr, wer_callback],
        verbose=args.verbose
    )

    # Save the model
    print("Training completed. Saving the model...")
    model.save('autocorrect_model.h5')

    # Save the vocabulary
    print("Saving vocabulary...")
    vocab_path = 'vocab.npz'
    np.savez(vocab_path, input_vocab=input_vocab, idx_to_char=idx_to_char)
    print("Vocabulary saved.")

    # Evaluation on the validation set
    print("\n=== Final Evaluation on Validation Set ===")

    # Convert all validation input texts to sequences
    val_input_seqs = np.array([text_to_sequence(text, max_seq_length, input_vocab) for text in val_input_texts])

    # Get predictions for the entire validation set
    predictions = model.predict(val_input_seqs, batch_size=args.batch_size)

    # Get the predicted indices
    predicted_indices = np.argmax(predictions, axis=-1)

    wers = []
    corrected_texts = []

    for idx in range(len(val_input_texts)):
        input_seq = val_input_seqs[idx]
        input_length = np.count_nonzero(input_seq)

        # Get the predicted indices for this sample
        predicted_seq = predicted_indices[idx][:input_length]

        # Convert indices to characters
        predicted_text = ''.join([idx_to_char.get(idx, '') for idx in predicted_seq])
        predicted_text = predicted_text.strip()
        corrected_texts.append(predicted_text)

        # Get the gold text
        gold_text = val_target_texts[idx]

        # Compute WER
        wer_score = jiwer.wer(gold_text, predicted_text)
        wers.append(wer_score)

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1}/{len(val_input_texts)} samples. Current WER: {wer_score:.4f}")

    average_wer = np.mean(wers) * 100
    print(f'Final Validation Average WER: {average_wer:.2f}%')

    # Save validation results
    print("Saving validation results...")
    val_results = pd.DataFrame({
        'corrupt_msg': val_input_texts,
        'gold_msg': val_target_texts,
        'corrected_msg': corrected_texts,
        'wer': wers
    })
    val_results.to_csv('validation_results.csv', index=False)
    print("Validation results saved to 'validation_results.csv'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character-Level Autocorrect Model Training Script")
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
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode')
    args = parser.parse_args()
    main(args)
