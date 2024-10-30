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

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)

        # Bidirectional GRU
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)

        self.fc = nn.Linear(hid_dim * 2, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src: [seq_len, batch_size]
        embedded = self.dropout(self.embedding(src))

        # embedded: [seq_len, batch_size, emb_dim]
        outputs, hidden = self.rnn(embedded)

        # outputs: [seq_len, batch_size, hid_dim * 2]
        # hidden: [n_layers * 2, batch_size, hid_dim]

        # Separate the hidden states for forward and backward passes
        # hidden_forward and hidden_backward: [n_layers, batch_size, hid_dim]
        hidden_forward = hidden[0:self.n_layers]
        hidden_backward = hidden[self.n_layers:self.n_layers*2]

        # Concatenate forward and backward hidden states for each layer
        hidden = torch.cat((hidden_forward, hidden_backward), dim=2)

        # hidden: [n_layers, batch_size, hid_dim * 2]

        # Pass through a linear layer and apply tanh activation
        hidden = torch.tanh(self.fc(hidden))

        # hidden: [n_layers, batch_size, hid_dim]
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super(Attention, self).__init__()

        self.attn = nn.Linear((hid_dim * 2) + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):

        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]
        src_len = encoder_outputs.shape[0]

        # Repeat hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)

        # hidden: [src_len, batch_size, hid_dim]
        encoder_outputs = encoder_outputs.permute(0, 1, 2)

        # Calculate energy
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy: [src_len, batch_size, hid_dim]
        attention = self.v(energy).squeeze(2)

        # attention: [src_len, batch_size]
        return nn.functional.softmax(attention.permute(1, 0), dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)

        self.rnn = nn.GRU((hid_dim * 2) + emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear((hid_dim * 2) + hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):

        # input: [batch_size]
        # hidden: [n_layers, batch_size, hid_dim]
        # encoder_outputs: [src_len, batch_size, hid_dim * 2]

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        # embedded: [1, batch_size, emb_dim]

        a = self.attention(hidden[-1], encoder_outputs)

        # a: [batch_size, src_len]
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # Weighted sum of encoder_outputs
        weighted = torch.bmm(a, encoder_outputs)

        # weighted: [batch_size, 1, hid_dim * 2]
        weighted = weighted.permute(1, 0, 2)

        # weighted: [1, batch_size, hid_dim * 2]
        rnn_input = torch.cat((embedded, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        # output: [1, batch_size, hid_dim]
        # hidden: [n_layers, batch_size, hid_dim]

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction: [batch_size, output_dim]
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg):

        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        encoder_outputs, hidden = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):

            output, hidden = self.decoder(input, hidden, encoder_outputs)

            outputs[t] = output

            teacher_force = np.random.rand() < self.teacher_forcing_ratio

            top1 = output.argmax(1)

            input = trg[t] if teacher_force else top1

        return outputs

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for src, trg in tqdm(iterator, desc="Training", leave=False):
        src = src.transpose(0, 1).to(model.device)
        trg = trg.transpose(0, 1).to(model.device)
        optimizer.zero_grad()
        output = model(src, trg)
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

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for src, trg in tqdm(iterator, desc="Evaluating", leave=False):
            src = src.transpose(0, 1).to(model.device)
            trg = trg.transpose(0, 1).to(model.device)
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    avg_loss = epoch_loss / len(iterator)
    return avg_loss

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

    # Hyperparameters
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 16
    TEACHER_FORCING_RATIO = 0.5
    N_EPOCHS = 20
    CLIP = 1

    # Initialize model
    attention = Attention(HID_DIM)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attention)
    model = Seq2Seq(encoder, decoder, device, TEACHER_FORCING_RATIO).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(N_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{N_EPOCHS} ===")
        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        val_loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model if validation loss decreases
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'seq2seq_model.pth')

    print("Training completed. Saving final model and vocabulary...")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Model and vocabulary saved.")

    # Evaluation on the validation set
    print("\n=== Final Evaluation on Validation Set ===")
    model.load_state_dict(torch.load('seq2seq_model.pth'))
    model.eval()
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
        encoder_outputs, hidden = model.encoder(src)
    input_token = torch.tensor([vocab['<sos>']]).to(device)
    outputs = []
    inv_vocab = {idx: ch for ch, idx in vocab.items()}
    for _ in range(max_length):
        with torch.no_grad():
            output, hidden = model.decoder(input_token, hidden, encoder_outputs)
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
        end_index=100,  # Adjust as needed
        max_length=100
    )
