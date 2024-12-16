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
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_bert_score_batch(self, candidates, references, lang='en'):
        references = [ref if ref.strip() != "" else " " for ref in references]
        P, R, F1 = bert_score.score(candidates, references, lang=lang, device=self.device, verbose=False)
        return F1.tolist()


def extract_gender_reviews(review_text):
    """
    Extract male and female-centered reviews from the given review text.

    """
    review_text = review_text.replace('“', '"').replace('”', '"')

    male_pattern = r'\{gender:\s*male\}\s*"([^"]+)"'
    female_pattern = r'\{gender:\s*female\}\s*"([^"]+)"'
    
    male_matches = re.findall(male_pattern, review_text, re.IGNORECASE | re.MULTILINE)
    female_matches = re.findall(female_pattern, review_text, re.IGNORECASE | re.MULTILINE)

    male_text = ' '.join([match.strip() for match in male_matches])
    female_text = ' '.join([match.strip() for match in female_matches])

    return male_text, female_text


def calculate_sof(male_scores, female_scores):
    male = np.array(male_scores)
    female = np.array(female_scores)
    sof = np.std(np.vstack([male, female]), axis=0)
    return sof.tolist()


def main(args):
    # Load the CSV file
    data = pd.read_csv(args.file_path)

    # Verify that the ID column exists
    if args.id_column not in data.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in the input CSV.")

    # Determine how many rows to process
    if args.num_rows is None:
        # By default, process all rows in the CSV
        num_rows = len(data)
    else:
        num_rows = min(args.num_rows, len(data))

    # Slice the data to process only the specified number of rows
    data = data.iloc[:num_rows]

    # Extract necessary columns
    ids = data[args.id_column].tolist()
    reviews = data[args.reviews_column].fillna("").tolist()

    print("Extracting male and female-centered reviews...")
    gender_reviews = [extract_gender_reviews(review) for review in tqdm(reviews, desc="Extracting Gender Reviews")]
    male_texts = [g[0] for g in gender_reviews]
    female_texts = [g[1] for g in gender_reviews]

    scorer = BERTScoreEvaluator(device=args.device)
    results = {"id": ids}

    for summary_col in args.summary_columns:
        print(f"\nProcessing summary column: {summary_col}")
        summaries = data[summary_col].fillna("").tolist()

        bertscore_male = []
        bertscore_female = []

        print("Calculating BERTScores for male-centered reviews...")
        for start in tqdm(range(0, len(summaries), args.batch_size), desc=f"BERTScore Male for {summary_col}"):
            end = start + args.batch_size
            batch_summaries = summaries[start:end]
            batch_male_refs = male_texts[start:end]
            batch_scores = scorer.calculate_bert_score_batch(batch_summaries, batch_male_refs)
            bertscore_male.extend(batch_scores)

        print("Calculating BERTScores for female-centered reviews...")
        for start in tqdm(range(0, len(summaries), args.batch_size), desc=f"BERTScore Female for {summary_col}"):
            end = start + args.batch_size
            batch_summaries = summaries[start:end]
            batch_female_refs = female_texts[start:end]
            batch_scores = scorer.calculate_bert_score_batch(batch_summaries, batch_female_refs)
            bertscore_female.extend(batch_scores)

        assert len(bertscore_male) == len(summaries), "Mismatch in bertscore_male length."
        assert len(bertscore_female) == len(summaries), "Mismatch in bertscore_female length."

        print("Calculating Overall BERTScore and SOF...")
        overall_bertscore = [(m + f) / 2.0 for m, f in zip(bertscore_male, bertscore_female)]
        sof = calculate_sof(bertscore_male, bertscore_female)

        results[f"{summary_col}_bertscore"] = overall_bertscore
        results[f"{summary_col}_sof"] = sof

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False)
    print(f"\nConsolidated results saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute row-wise BERTScore and SOF for male and female-centered reviews.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--id_column", type=str, default="id", help="Column name for the unique identifier (default: 'id').")
    parser.add_argument("--reviews_column", type=str, required=True, help="Column name for reviews.")
    parser.add_argument("--summary_columns", type=str, nargs='+', required=True, help="List of column names for summaries.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the consolidated results CSV file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--num_rows", type=int, default=None, help="Number of rows to process (default: all rows).")

    args = parser.parse_args()
    main(args)
