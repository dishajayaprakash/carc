mport pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
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
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
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
            input = trg[t] if np.random.rand() < teacher_forcing_ratio else top1
        return outputs

def train(model, iterator, optimizer, criterion, clip, epoch):
    model.train()
    epoch_loss = 0
    print(f"Starting training for epoch {epoch+1}...")
    for i, (src, trg) in enumerate(iterator):
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

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}], Batch [{i+1}/{len(iterator)}], Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(iterator)
    print(f"Epoch [{epoch+1}] Training Completed. Average Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, iterator, criterion, target_vocab, epoch):
    model.eval()
    epoch_loss = 0
    wers = []
    inv_target_vocab = {idx: ch for ch, idx in target_vocab.items()}
    print(f"Starting evaluation for epoch {epoch+1}...")

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.transpose(0, 1).to(model.device)
            trg = trg.transpose(0, 1).to(model.device)
            output = model(src, trg, teacher_forcing_ratio=0)  # No teacher forcing
            output_dim = output.shape[-1]
            output_loss = output[1:].view(-1, output_dim)
            trg_loss = trg[1:].reshape(-1)
            loss = criterion(output_loss, trg_loss)
            epoch_loss += loss.item()

            # Calculate WER
            output_tokens = output.argmax(2)  # [seq_len, batch_size]
            for idx in range(output_tokens.shape[1]):
                pred_seq = output_tokens[:, idx].cpu().numpy()
                trg_seq = trg[:, idx].cpu().numpy()

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

def translate_sentence(model, sentence, input_vocab, target_vocab, max_length=100):
    model.eval()
    tokens = [input_vocab.get(ch, input_vocab['<unk>']) for ch in list(sentence.lower())]
    src = torch.tensor([input_vocab['<sos>']] + tokens + [input_vocab['<eos>']]).unsqueeze(1).to(model.device)
    hidden = model.encoder(src)
    input_token = torch.tensor([target_vocab['<sos>']]).to(model.device)
    outputs = []
    for t in range(max_length):
        output, hidden = model.decoder(input_token, hidden)
        top1 = output.argmax(1)
        if top1.item() == target_vocab['<eos>']:
            break
        else:
            outputs.append(top1.item())
            input_token = top1
    inv_target_vocab = {idx: ch for ch, idx in target_vocab.items()}
    translated_sentence = ''.join([inv_target_vocab.get(idx, '') for idx in outputs])
    return translated_sentence

def main(train_filename, validation_filename, start_index=None, end_index=None, num_epochs=10, batch_size=64, learning_rate=0.001, emb_dim=256, hid_dim=512, n_layers=2, dropout=0.5, max_length=100):
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

    print("Building vocabularies...")
    # Build vocabularies
    input_vocab = build_char_vocab(train_input_texts + val_input_texts)
    target_vocab = build_char_vocab(train_target_texts + val_target_texts)

    print("Creating datasets and dataloaders...")
    # Create datasets and dataloaders
    train_dataset = AutocorrectDataset(train_input_texts, train_target_texts, input_vocab, target_vocab, max_length)
    val_dataset = AutocorrectDataset(val_input_texts, val_target_texts, input_vocab, target_vocab, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_dim = len(input_vocab)
    output_dim = len(target_vocab)
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    print("Model initialized.")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> token

    # Training loop
    best_val_wer = float('inf')
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
        train_loss = train(model, train_loader, optimizer, criterion, clip=1, epoch=epoch)
        val_loss, val_wer = evaluate(model, val_loader, criterion, target_vocab, epoch=epoch)
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val WER: {val_wer:.4f}")

        # Save the best model based on WER
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            torch.save(model.state_dict(), 'best_seq2seq_model.pth')
            print(f"Best model saved with Val WER: {best_val_wer:.4f}")

    # Save the final model and vocabularies
    print("Training completed. Saving final model and vocabularies...")
    torch.save(model.state_dict(), 'seq2seq_model.pth')
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
        prediction = translate_sentence(model, input_text, input_vocab, target_vocab, max_length)
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

if __name__ == "__main__":
    train_csv_path = 'train_fold.csv'
    validation_csv_path = 'val_fold.csv'

    main(
        train_filename=train_csv_path,
        validation_filename=validation_csv_path,
        start_index=0,
        end_index=180000,  # Adjust as needed
        num_epochs=40,
        batch_size=64,
        learning_rate=0.005,
        emb_dim=256,
        hid_dim=512,
        n_layers=3,
        dropout=0.5,
        max_length=100
    )