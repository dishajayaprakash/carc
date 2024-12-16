import pandas as pd
import argparse
from transformers import AutoModel, AutoTokenizer
import bert_score
import numpy as np
import warnings
import re
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class BERTScoreEvaluator:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        """
        Initializes the BERTScore evaluator.

        Args:
            model_name (str): Name of the BERT model to use.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_bert_score_batch(self, candidates, references, lang='en'):
        """
        Compute BERTScore in batches.

        Args:
            candidates (list of str): List of candidate summaries.
            references (list of str): List of reference texts (pos, neg, neu).
            lang (str): Language code.

        Returns:
            list of float: List of BERTScore F1 for each pair.
        """
        # Handle empty references by replacing them with a space
        references = [ref if ref.strip() != "" else " " for ref in references]
        # Compute BERTScore
        P, R, F1 = bert_score.score(candidates, references, lang=lang, device=self.device, verbose=False)
        return F1.tolist()


def extract_sentiments(review_text):
    """
    Extracts and concatenates all sentences with the specified sentiment from a review.

    Args:
        review_text (str): The review text containing sentiment tags.

    Returns:
        tuple: (pos_text, neg_text, neu_text)
    """
    review_text = review_text.replace('“', '"').replace('”', '"')

    # Define regex patterns for each sentiment
    pos_pattern = r'\{sentiment:\s*pos\}\s*"([^"]+)"'
    neg_pattern = r'\{sentiment:\s*neg\}\s*"([^"]+)"'
    neu_pattern = r'\{sentiment:\s*neu\}\s*"([^"]+)"'


    # Find all matches for each sentiment
    pos_matches = re.findall(pos_pattern, review_text, re.DOTALL | re.IGNORECASE)
    neg_matches = re.findall(neg_pattern, review_text, re.DOTALL | re.IGNORECASE)
    neu_matches = re.findall(neu_pattern, review_text, re.DOTALL | re.IGNORECASE)

    # Concatenate all matched texts for each sentiment
    pos_text = ' '.join([match.strip() for match in pos_matches])
    neg_text = ' '.join([match.strip() for match in neg_matches])
    neu_text = ' '.join([match.strip() for match in neu_matches])

    return pos_text, neg_text, neu_text


def calculate_sof(pos_scores, neg_scores, neu_scores):
    """
    Calculate SOF (Standard Deviation of BERTScores) for each row.

    Args:
        pos_scores (list of float): BERTScore F1 for positive sentiment.
        neg_scores (list of float): BERTScore F1 for negative sentiment.
        neu_scores (list of float): BERTScore F1 for neutral sentiment.

    Returns:
        list of float: SOF for each row.
    """
    # Convert lists to numpy arrays for vectorized operations
    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    neu = np.array(neu_scores)
    # Stack the scores vertically and compute std dev across axis=0 (per row)
    sof = np.std(np.vstack([pos, neg, neu]), axis=0)
    return sof.tolist()


def main(args):
    # Load the CSV file
    data = pd.read_csv(args.file_path)

    # Verify that the ID column exists
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in the input CSV.")

    # Extract necessary columns
    ids = data[args.id_column].tolist()
    reviews = data[args.reviews_column].fillna("").tolist()

    # Extract sentiments for each row
    print("Extracting sentiments from reviews...")
    sentiments = [extract_sentiments(review) for review in tqdm(reviews, desc="Extracting Sentiments")]
    pos_texts = [s[0] for s in sentiments]
    neg_texts = [s[1] for s in sentiments]
    neu_texts = [s[2] for s in sentiments]

    # Initialize the BERTScore evaluator
    scorer = BERTScoreEvaluator(device=args.device)

    # Initialize the results dictionary with IDs
    results = {"id": ids}

    # Iterate over each summary column
    for summary_col in args.summary_columns:
        print(f"\nProcessing summary column: {summary_col}")
        summaries = data[summary_col].fillna("").tolist()

        bertscore_pos = []
        bertscore_neg = []
        bertscore_neu = []

        # Process in batches for positive sentiments
        print("Calculating BERTScores for positive sentiments...")
        for start in tqdm(range(0, len(summaries), args.batch_size), desc=f"BERTScore Pos for {summary_col}"):
            end = start + args.batch_size
            batch_summaries = summaries[start:end]
            batch_pos_refs = pos_texts[start:end]
            batch_scores = scorer.calculate_bert_score_batch(batch_summaries, batch_pos_refs)
            bertscore_pos.extend(batch_scores)

        # Process in batches for negative sentiments
        print("Calculating BERTScores for negative sentiments...")
        for start in tqdm(range(0, len(summaries), args.batch_size), desc=f"BERTScore Neg for {summary_col}"):
            end = start + args.batch_size
            batch_summaries = summaries[start:end]
            batch_neg_refs = neg_texts[start:end]
            batch_scores = scorer.calculate_bert_score_batch(batch_summaries, batch_neg_refs)
            bertscore_neg.extend(batch_scores)

        # Process in batches for neutral sentiments
        print("Calculating BERTScores for neutral sentiments...")
        for start in tqdm(range(0, len(summaries), args.batch_size), desc=f"BERTScore Neu for {summary_col}"):
            end = start + args.batch_size
            batch_summaries = summaries[start:end]
            batch_neu_refs = neu_texts[start:end]
            batch_scores = scorer.calculate_bert_score_batch(batch_summaries, batch_neu_refs)
            bertscore_neu.extend(batch_scores)

        # Ensure all bertscore lists have the same length
        assert len(bertscore_pos) == len(summaries), "Mismatch in bertscore_pos length."
        assert len(bertscore_neg) == len(summaries), "Mismatch in bertscore_neg length."
        assert len(bertscore_neu) == len(summaries), "Mismatch in bertscore_neu length."

        # Compute overall BERTScore and SOF for each row
        print("Calculating Overall BERTScore and SOF...")
        overall_bertscore = [(p + n + neu) / 3.0 for p, n, neu in zip(bertscore_pos, bertscore_neg, bertscore_neu)]
        sof = calculate_sof(bertscore_pos, bertscore_neg, bertscore_neu)

        # Add the scores to the results dictionary
        results[f"{summary_col}_bertscore"] = overall_bertscore
        results[f"{summary_col}_sof"] = sof

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Save the results to a CSV file
    results_df.to_csv(args.output_file, index=False)
    print(f"\nConsolidated results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute row-wise BERTScore and SOF for multiple summary columns.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--id_column", type=str, default="id", help="Column name for the unique identifier (default: 'id').")
    parser.add_argument("--reviews_column", type=str, required=True, help="Column name for reviews.")
    parser.add_argument("--summary_columns", type=str, nargs='+', required=True, help="List of column names for summaries.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the consolidated results CSV file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()
    main(args)
