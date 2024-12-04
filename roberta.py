import pandas as pd
import argparse
from transformers import AutoModel, AutoTokenizer
import bert_score

class GenderRepresentationBERTScore:
    def __init__(self, model_name="bert-base-uncased", device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def calculate_bert_score_batch(self, candidates, references, lang='en'):
        """
        Compute BERTScore in batches.
        Args:
            candidates (list of str): List of candidate texts.
            references (list of str): List of reference texts.
            lang (str): Language code.
        Returns:
            list of float: List of BERTScore F1 for each pair.
        """
        _, _, F1 = bert_score.score(candidates, references, lang=lang, device=self.device, verbose=False)
        return F1.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BERTScore for reviews vs summaries.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--reviews_column", type=str, required=True, help="Column name for reviews.")
    parser.add_argument("--summary_column", type=str, required=True, help="Column name for summaries.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results CSV.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda' or 'cpu').")

    args = parser.parse_args()

    # Load the CSV file
    data = pd.read_csv(args.file_path)

    # Extract reviews and summaries as lists
    reviews = data[args.reviews_column].tolist()
    summaries = data[args.summary_column].tolist()

    # Initialize the scorer
    gender_scorer = GenderRepresentationBERTScore(device=args.device)

    # Batch size for processing
    batch_size = args.batch_size
    overall_bert_scores = []

    # Process in batches
    for start in range(0, len(reviews), batch_size):
        end = start + batch_size
        batch_reviews = reviews[start:end]
        batch_summaries = summaries[start:end]
        batch_scores = gender_scorer.calculate_bert_score_batch(batch_reviews, batch_summaries)
        overall_bert_scores.extend(batch_scores)
        print(f"Processed {end}/{len(reviews)} rows.")

    # Prepare results
    results = pd.DataFrame({
        "id": data["id"],
        "Overall BERTScore F1": overall_bert_scores
    })

    # Save results to CSV
    results.to_csv(args.output_path, index=False)

    print(f"Results saved to {args.output_path}")
