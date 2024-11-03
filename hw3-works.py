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
import matplotlib.pyplot as plt
import sys

def main():
    # ------------------------------
    # Parse Command-Line Arguments
    # ------------------------------
    parser = argparse.ArgumentParser(description='Character-Level Autocorrect Model')

    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of embedding layer')
    parser.add_argument('--lstm_units', type=int, default=64, help='Number of LSTM units')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to read from the dataset')
    parser.add_argument('--train_data', type=str, default='train_fold.csv', help='Path to training data CSV file')
    parser.add_argument('--val_data', type=str, default='val_fold.csv', help='Path to validation data CSV file')
    parser.add_argument('--verbose', type=int, default=2, help='Verbosity mode')

    args = parser.parse_args()

    # ------------------------------
    # 1. Load and Preprocess the Data
    # ------------------------------

    # Load the training dataset
    df_train = pd.read_csv(args.train_data, nrows=args.max_samples)
    df_val = pd.read_csv(args.val_data, nrows=args.max_samples)

    # Preprocess the data: lowercase and remove punctuation
    def preprocess_text(text):
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    df_train['corrupt_msg'] = df_train['corrupt_msg'].astype(str).apply(preprocess_text)
    df_train['gold_msg'] = df_train['gold_msg'].astype(str).apply(preprocess_text)

    df_val['corrupt_msg'] = df_val['corrupt_msg'].astype(str).apply(preprocess_text)
    df_val['gold_msg'] = df_val['gold_msg'].astype(str).apply(preprocess_text)

    # Display first few entries to verify
    print("Sample Training Data:")
    print(df_train.head())

    print("\nSample Validation Data:")
    print(df_val.head())

    # ------------------------------
    # Extract Texts
    # ------------------------------

    X_train_texts = df_train['corrupt_msg']
    y_train_texts = df_train['gold_msg']

    X_val_texts = df_val['corrupt_msg']
    y_val_texts = df_val['gold_msg']

    print(f"\nTraining samples: {len(X_train_texts)}")
    print(f"Validation samples: {len(X_val_texts)}")

    # ------------------------------
    # 3. Build Character Mappings
    # ------------------------------

    # Extract unique characters from both corrupt and gold messages
    all_chars = set(''.join(X_train_texts) + ''.join(y_train_texts))
    ALL_CHARS = sorted(list(all_chars))

    # Assign unique indices to each character
    char_to_idx = {char: idx + 1 for idx, char in enumerate(ALL_CHARS)}
    char_to_idx['<PAD>'] = 0  # Padding character
    char_to_idx['<UNK>'] = len(char_to_idx) + 1  # Unknown character

    # Create inverse mapping
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}

    VOCAB_SIZE = len(char_to_idx)
    print(f"\nVocabulary size (including <PAD> and <UNK>): {VOCAB_SIZE}")

    # ------------------------------
    # 4. Creating Input and Target Sequences
    # ------------------------------

    # Determine maximum sequence length
    MAX_SEQ_LENGTH = max(
        max(len(text) for text in X_train_texts),
        max(len(text) for text in y_train_texts),
        max(len(text) for text in X_val_texts),
        max(len(text) for text in y_val_texts)
    )

    print(f"Max sequence length: {MAX_SEQ_LENGTH}")

    # Define text to sequence function
    def text_to_sequence(text, max_length):
        seq = [char_to_idx.get(char, char_to_idx['<UNK>']) for char in text]
        return pad_sequences([seq], maxlen=max_length, padding='post', value=char_to_idx['<PAD>'])[0]

    # Convert texts to sequences
    X_train = np.array([text_to_sequence(text, MAX_SEQ_LENGTH) for text in X_train_texts])
    X_val = np.array([text_to_sequence(text, MAX_SEQ_LENGTH) for text in X_val_texts])

    y_train = np.array([text_to_sequence(text, MAX_SEQ_LENGTH) for text in y_train_texts])
    y_val = np.array([text_to_sequence(text, MAX_SEQ_LENGTH) for text in y_val_texts])

    # Expand dimensions for y to match the required shape
    y_train = np.expand_dims(y_train, -1)
    y_val = np.expand_dims(y_val, -1)

    print(f"X_train shape: {X_train.shape}")  # (num_samples, MAX_SEQ_LENGTH)
    print(f"y_train shape: {y_train.shape}")  # (num_samples, MAX_SEQ_LENGTH, 1)

    # Create sample weights to mask the loss over padded positions
    sample_weights_train = np.where(y_train.squeeze(-1) == char_to_idx['<PAD>'], 0.0, 1.0)
    sample_weights_val = np.where(y_val.squeeze(-1) == char_to_idx['<PAD>'], 0.0, 1.0)

    # ------------------------------
    # 5. Defining the Encoder-Only Model
    # ------------------------------

    # Clear any existing models
    tf.keras.backend.clear_session()

    # Define model parameters from command-line arguments
    EMBEDDING_DIM = args.embedding_dim
    LSTM_UNITS = args.lstm_units
    DROPOUT_RATE = args.dropout_rate

    # Model
    inputs = Input(shape=(MAX_SEQ_LENGTH,), name='input')
    embedding = Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True,
        name='embedding'
    )(inputs)

    # Apply dropout after embedding
    embedding = Dropout(DROPOUT_RATE, name='embedding_dropout')(embedding)

    encoder_outputs = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=True),
        name='bidirectional_lstm'
    )(embedding)

    # Apply dropout after LSTM
    encoder_outputs = Dropout(DROPOUT_RATE, name='lstm_dropout')(encoder_outputs)

    outputs = TimeDistributed(
        Dense(VOCAB_SIZE, activation='softmax'),
        name='time_distributed_dense'
    )(encoder_outputs)

    # Define the model
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # ------------------------------
    # 6. Implementing Callbacks
    # ------------------------------

    # Define Early Stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Define Learning Rate Reduction
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Define WER Callback
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
                # Get the predicted indices and the input length
                predicted_seq = predicted_indices[i]
                input_seq = X_val[i]
                # Compute the input length (number of non-zero tokens in input)
                input_length = np.count_nonzero(input_seq)
                # Limit the predicted sequence to the input length
                predicted_seq = predicted_seq[:input_length]
                # Convert indices to characters
                predicted_text = ''.join([self.idx_to_char.get(idx, '') for idx in predicted_seq])
                predicted_text = predicted_text.strip()
                predicted_texts.append(predicted_text)

                # Get the gold text
                gold_seq = y_val[i].squeeze()
                gold_seq = gold_seq[:input_length]
                gold_text = ''.join([self.idx_to_char.get(idx, '') for idx in gold_seq])
                gold_text = gold_text.strip()
                reference_texts.append(gold_text)

            # Calculate WER
            wer_scores = [jiwer.wer(ref, hyp) for ref, hyp in zip(reference_texts, predicted_texts)]
            average_wer = np.mean(wer_scores) * 100  # Convert to percentage

            print(f"Epoch {epoch+1} - Validation WER: {average_wer:.2f}%")

    # Instantiate the WER callback
    wer_callback = WERCallback(validation_data=(X_val, y_val), idx_to_char=idx_to_char)

    # ------------------------------
    # 7. Training the Model
    # ------------------------------

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val, sample_weights_val),
        callbacks=[early_stopping, reduce_lr, wer_callback],
        verbose=args.verbose
    )

    # ------------------------------
    # 8. Inference Function
    # ------------------------------

    def decode_sequence(input_seq):
        """
        Decode the input sequence to generate the corrected text.
        """
        predictions = model.predict(input_seq)
        predicted_indices = np.argmax(predictions, axis=-1)[0]

        # Compute input length (number of non-zero tokens in input)
        input_length = np.count_nonzero(input_seq)

        # Limit the predicted indices to the input_length
        predicted_indices = predicted_indices[:input_length]

        # Convert indices to characters
        predicted_text = ''.join([idx_to_char.get(idx, '') for idx in predicted_indices])
        predicted_text = predicted_text.strip()
        return predicted_text

    # ------------------------------
    # 9. Testing the Model
    # ------------------------------

    # Select a few samples from validation data to test
    for seq_index in range(5):
        # Take one sequence from the validation set
        input_text = X_val_texts.iloc[seq_index]
        input_seq = text_to_sequence(input_text, MAX_SEQ_LENGTH)
        input_seq = input_seq.reshape(1, -1)
        decoded_sentence = decode_sequence(input_seq)

        # Get the gold (actual) text
        gold_text = y_val_texts.iloc[seq_index]

        print("Input (Corrupt):", input_text)
        print("Predicted Correction:", decoded_sentence)
        print("Actual Correction:", gold_text)
        print('-'*50)

    # ------------------------------
    # 10. Optional: Plot Training History
    # ------------------------------

    # Uncomment the following lines if you wish to plot the training history
    # plt.figure(figsize=(12, 4))
    #
    # # Plot loss
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['loss'], label='Training Loss', color='b')
    # plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    # plt.title('Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # # Plot accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['accuracy'], label='Training Accuracy', color='b')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    # plt.title('Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.show()

if __name__ == '__main__':
    main()
