import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from jiwer import wer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import itertools

def load_data(filename, start_index=None, end_index=None):
    print(f"Loading data from {filename}...")
    data = pd.read_csv(filename)
    if start_index is not None and end_index is not None:
        data = data.iloc[start_index:end_index]
        print(f"Data sliced from index {start_index} to {end_index}")
    print(f"Total samples loaded: {len(data)}")
    return data

def tokenize(text):
    return list(text.lower())

class AutocorrectDataset(Dataset):
    def __init__(self, input_texts, target_texts, vocab, max_length=100):
        self.input_texts = input_texts
        self.target_texts = target_texts
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_seq = [self.vocab['<sos>']] + [self.vocab.get(ch, self.vocab['<unk>']) for ch in self.input_texts[idx]] + [self.vocab['<eos>']]
        target_seq = [self.vocab['<sos>']] + [self.vocab.get(ch, self.vocab['<unk>']) for ch in self.target_texts[idx]] + [self.vocab['<eos>']]

        # Truncate sequences if they exceed max_length
        input_seq = input_seq[:self.max_length]
        target_seq = target_seq[:self.max_length]

        # Padding
        input_seq += [self.vocab['<pad>']] * (self.max_length - len(input_seq))
        target_seq += [self.vocab['<pad>']] * (self.max_length - len(target_seq))

        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        return input_seq, target_seq

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

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [seq_len, batch_size]
        embedded = self.dropout(self.embedding(src))  # [seq_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)          # outputs: [seq_len, batch_size, hid_dim]
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        # input: [batch_size]
        input = input.unsqueeze(0)  # [1, batch_size]
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, emb_dim]
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg):
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        hidden = self.encoder(src)

        input = trg[0, :]  # Start with <sos>
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.argmax(1)
            teacher_force = np.random.rand() < self.teacher_forcing_ratio
            input = trg[t] if teacher_force else top1
        return outputs

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(iterator, desc="Training", leave=False):
        src = src.transpose(0, 1).to(model.device)  # [seq_len, batch_size]
        trg = trg.transpose(0, 1).to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)
        # Exclude the first token (<sos>) from loss calculation
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(iterator)
    return avg_loss

def evaluate(model, iterator, criterion, vocab):
    model.eval()
    epoch_loss = 0
    inv_vocab = {idx: ch for ch, idx in vocab.items()}

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating", leave=False):
            src = src.transpose(0, 1).to(model.device)
            trg = trg.transpose(0, 1).to(model.device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output_loss = output[1:].view(-1, output_dim)
            trg_loss = trg[1:].reshape(-1)
            loss = criterion(output_loss, trg_loss)
            epoch_loss += loss.item()
    avg_loss = epoch_loss / len(iterator)
    return avg_loss

def grid_search(train_dataset, val_dataset, vocab, device, hyperparameter_grid):
    print("Starting grid search over hyperparameters...")
    results = []
    for params in hyperparameter_grid:
        emb_dim = params['emb_dim']
        hid_dim = params['hid_dim']
        n_layers = params['n_layers']
        dropout = params['dropout']
        learning_rate = params['learning_rate']
        batch_size = params['batch_size']
        teacher_forcing_ratio = params['teacher_forcing_ratio']

        print(f"\nTesting hyperparameters: {params}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = len(vocab)
        output_dim = len(vocab)
        encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
        decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
        model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])  # Ignore <pad> token

        # Train for one epoch
        train_loss = train(model, train_loader, optimizer, criterion, clip=1)
        val_loss = evaluate(model, val_loader, criterion, vocab)

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save results
        results.append({
            'params': params,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # Find the best hyperparameters based on validation loss
    best_result = min(results, key=lambda x: x['val_loss'])
    print(f"\nBest hyperparameters based on validation loss:")
    print(best_result['params'])
    print(f"Train Loss: {best_result['train_loss']:.4f}, Val Loss: {best_result['val_loss']:.4f}")
    return best_result['params']

def main(train_filename, validation_filename, start_index=None, end_index=None, max_length=100):
    print("Initializing training process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data
    train_data = load_data(train_filename, start_index, end_index)
    val_data = load_data(validation_filename)
    train_input_texts = train_data['corrupt_msg'].tolist()
    train_target_texts = train_data['gold_msg'].tolist()
    val_input_texts = val_data['corrupt_msg'].tolist()
    val_target_texts = val_data['gold_msg'].tolist()

    print("Building vocabulary...")
    # Build a combined vocabulary
    all_texts = train_input_texts + train_target_texts + val_input_texts + val_target_texts
    vocab = build_char_vocab(all_texts)

    print("Creating datasets...")
    # Create datasets
    train_dataset = AutocorrectDataset(train_input_texts, train_target_texts, vocab, max_length)
    val_dataset = AutocorrectDataset(val_input_texts, val_target_texts, vocab, max_length)

    # Define hyperparameter grid
    emb_dim_list = [128, 256]
    hid_dim_list = [256, 512]
    n_layers_list = [1, 2]
    dropout_list = [0.3, 0.5]
    learning_rate_list = [0.001, 0.0005]
    batch_size_list = [64, 128]
    teacher_forcing_ratio_list = [0.5, 0.7]

    hyperparameter_grid = [
        {
            'emb_dim': emb_dim,
            'hid_dim': hid_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'teacher_forcing_ratio': teacher_forcing_ratio
        }
        for emb_dim in emb_dim_list
        for hid_dim in hid_dim_list
        for n_layers in n_layers_list
        for dropout in dropout_list
        for learning_rate in learning_rate_list
        for batch_size in batch_size_list
        for teacher_forcing_ratio in teacher_forcing_ratio_list
    ]

    # Limit the number of combinations for practicality
    max_combinations = 10
    hyperparameter_grid = hyperparameter_grid[:max_combinations]

    # Perform grid search
    best_params = grid_search(train_dataset, val_dataset, vocab, device, hyperparameter_grid)

    # Train final model with best hyperparameters
    print("\nTraining final model with best hyperparameters...")
    emb_dim = best_params['emb_dim']
    hid_dim = best_params['hid_dim']
    n_layers = best_params['n_layers']
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    teacher_forcing_ratio = best_params['teacher_forcing_ratio']
    num_epochs = 10  # Adjust as needed

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = len(vocab)
    output_dim = len(vocab)
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device, teacher_forcing_ratio).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])  # Ignore <pad> token

    best_val_wer = float('inf')
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        train_loss = train(model, train_loader, optimizer, criterion, clip=1)
        val_loss = evaluate(model, val_loader, criterion, vocab)
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the final model and vocabulary
    print("Training completed. Saving final model and vocabulary...")
    torch.save(model.state_dict(), 'seq2seq_model.pth')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Model and vocabulary saved.")

    # Evaluation on the validation set
    print("\n=== Final Evaluation on Validation Set ===")
    wers = []
    corrected_texts = []
    inv_vocab = {idx: ch for ch, idx in vocab.items()}
    for idx, (input_text, target_text) in enumerate(tqdm(zip(val_input_texts, val_target_texts), total=len(val_input_texts), desc="Validation Inference")):
        prediction = translate_sentence(model, input_text, vocab, device, max_length)
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

def translate_sentence(model, sentence, vocab, device, max_length=100):
    model.eval()
    tokens = [vocab.get(ch, vocab['<unk>']) for ch in list(sentence.lower())]
    src = torch.tensor([vocab['<sos>']] + tokens + [vocab['<eos>']]).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden = model.encoder(src)
    input_token = torch.tensor([vocab['<sos>']]).to(device)
    outputs = []
    inv_vocab = {idx: ch for ch, idx in vocab.items()}
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden)
            top1 = output.argmax(1)
        if top1.item() == vocab['<eos>']:
            break
        else:
            outputs.append(top1.item())
            input_token = top1
    translated_sentence = ''.join([inv_vocab.get(idx, '') for idx in outputs])
    return translated_sentence

if __name__ == "__main__":
    # Replace the file paths below with your actual file paths.
    train_csv_path = 'train_fold.csv'
    validation_csv_path = 'val_fold.csv'

    main(
        train_filename=train_csv_path,
        validation_filename=validation_csv_path,
        start_index=0,
        end_index=180000,  # Adjust as needed
        max_length=100
    )
