import os
import re
import string
import warnings
import logging
import pickle
import zipfile
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_scheduler,
)
from sklearn.metrics import classification_report
from tqdm import tqdm

# Global configurations
warnings.filterwarnings("ignore")
nltk.download("stopwords")
nltk.download("punkt")

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Utility functions
def setup_kaggle_api():
    """Sets up Kaggle API credentials."""
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(
            """
            {
              "username": "ishaanthanekar",
              "key": "44a843b1f5dad3498eb440c714fa29c3"
            }
            """
        )
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    os.system('kaggle datasets download -d ishaanthanekar/multilingual-reviews')


def extract_dataset(dataset_path, extract_path):
    """Extracts the dataset from a zip file."""
    with zipfile.ZipFile(dataset_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)


def preprocess_text(text, stopwords_set):
    """Cleans and tokenizes the text."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_set]
    return " ".join(tokens)


def preprocess_labels(data):
    """Maps star ratings to sentiment labels."""
    data["sentiment"] = data["stars"].apply(
        lambda x: 0 if x in [1, 2] else (1 if x == 3 else 2)
    )
    return data["cleaned_text"].tolist(), data["sentiment"].tolist()


# Dataset classes
class SentimentDataset(Dataset):
    """Dataset class for tokenized data."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Model training and evaluation functions
def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=4
):
    """Trains the Roberta model."""
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_correct = 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch["labels"]).sum().item()

            progress_bar.set_postfix(
                loss=total_loss / len(train_loader),
                accuracy=total_correct / len(train_loader.dataset),
            )

        logger.info(
            f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f} | "
            f"Accuracy: {total_correct / len(train_loader.dataset):.4f}"
        )
        evaluate_model(model, val_loader, device, "Validation")


def evaluate_model(model, data_loader, device, split_name):
    """Evaluates the model on the given dataset."""
    model.eval()
    total_loss, total_correct = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch["labels"]).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    logger.info(f"{split_name} Loss: {total_loss / len(data_loader):.4f}")
    logger.info(f"{split_name} Accuracy: {total_correct / len(data_loader.dataset):.4f}")
    logger.info(f"{split_name} Classification Report:")
    logger.info(
        classification_report(
            all_labels, all_preds, target_names=["Negative", "Neutral", "Positive"]
        )
    )


# Main script
def main():
    # Paths
    dataset_path = "multilingual-reviews.zip"
    extract_path = "dataset"

    # Setup
    setup_kaggle_api()
    extract_dataset(dataset_path, extract_path)

    # Load datasets
    train_df = pd.read_csv(f"{extract_path}/train.csv")
    validation_df = pd.read_csv(f"{extract_path}/validation.csv")
    test_df = pd.read_csv(f"{extract_path}/test.csv")

    # Filter languages
    selected_languages = ["en", "fr", "es"]
    train_df = train_df[train_df["language"].isin(selected_languages)]
    validation_df = validation_df[validation_df["language"].isin(selected_languages)]
    test_df = test_df[test_df["language"].isin(selected_languages)]

    # Preprocess data
    stop_words = (
        set(stopwords.words("english"))
        .union(set(stopwords.words("spanish")))
        .union(set(stopwords.words("french")))
    )
    train_df["cleaned_text"] = train_df["review_body"].apply(
        lambda x: preprocess_text(x, stop_words)
    )
    validation_df["cleaned_text"] = validation_df["review_body"].apply(
        lambda x: preprocess_text(x, stop_words)
    )
    test_df["cleaned_text"] = test_df["review_body"].apply(
        lambda x: preprocess_text(x, stop_words)
    )

    # Preprocess labels
    train_texts, train_labels = preprocess_labels(train_df)
    val_texts, val_labels = preprocess_labels(validation_df)
    test_texts, test_labels = preprocess_labels(test_df)

    # Dataset and Dataloader
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model setup
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * 4  # Assuming 4 epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Train and evaluate
    train_model(model, train_loader, val_loader, optimizer, scheduler, device)
    evaluate_model(model, test_loader, device, "Test")


if __name__ == "__main__":
    main()
