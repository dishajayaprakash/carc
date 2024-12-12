import openai
import os
import pandas as pd
import json
import requests
import argparse

def initialSumm(df, model, max_tokens, base_url, headers):
    df["summary1"] = [""] * len(df)
    for i in range(len(df)):
        prompt = """Below is an instruction that describes a task.
### Instruction: You are given a set of reviews separated by ||. All reviews are in the format {rating: NUMBER} \u201c{REVIEW_TEXT}\u201c, including rating information of a product given by the review writer. Please write a short text containing the salient information, i.e. a summary.

Reviews: """ + df['reviews'][i] + """\n\nsummary:"""

        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }

        response = requests.post(f"{base_url}/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            summary = response_data.get("choices", [{}])[0].get("text", "").strip()
            df.loc[i, "summary1"] = summary
        else:
            print(f"Error for row {i}: {response.status_code}, {response.text}")
            df.loc[i, "summary1"] = None

    return df

def generateFeedback(df, summary_col, feedback_col, model, max_tokens, base_url, headers):
    df[feedback_col] = [""] * len(df)
    for i in range(len(df)):
        prompt = """You are an annotator for the fairness of summarization...""" + df["reviews"][i] + """\n\nsummary: """ + df[summary_col][i] + """\n\nfeedback:"""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        response = requests.post(f"{base_url}/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            feedback = response_data.get("choices", [{}])[0].get("text", "").strip()
            df.loc[i, feedback_col] = feedback
        else:
            print(f"Error for row {i}: {response.status_code}, {response.text}")
            df.loc[i, feedback_col] = None

    return df

def refineWFeedback(df, summary_col, feedback_col, refined_summary_col, model, max_tokens, base_url, headers):
    df[refined_summary_col] = [""] * len(df)
    for i in range(len(df)):
        prompt = """Below is an instruction that describes a task...""" + df["reviews"][i] + """\n\nsummary: """ + df[summary_col][i] + """\n\nfeedback: """ + df[feedback_col][i] + """\n\nRefined summary:"""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        response = requests.post(f"{base_url}/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            refined_summary = response_data.get("choices", [{}])[0].get("text", "").strip()
            df.loc[i, refined_summary_col] = refined_summary
        else:
            print(f"Error for row {i}: {response.status_code}, {response.text}")
            df.loc[i, refined_summary_col] = None

    return df

def refineWOFeedback(df, summary_col, refined_summary_col, model, max_tokens, base_url, headers):
    df[refined_summary_col] = [""] * len(df)
    for i in range(len(df)):
        prompt = """Below is an instruction that describes a task...""" + df["reviews"][i] + """\n\nsummary: """ + df[summary_col][i] + """\n\nRefined summary:"""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        response = requests.post(f"{base_url}/completions", headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            refined_summary = response_data.get("choices", [{}])[0].get("text", "").strip()
            df.loc[i, refined_summary_col] = refined_summary
        else:
            print(f"Error for row {i}: {response.status_code}, {response.text}")
            df.loc[i, refined_summary_col] = None

    return df

def summary_refinement_W_feedback(df, model, max_tokens, base_url, headers):
    df = initialSumm(df, model, max_tokens, base_url, headers)
    df = generateFeedback(df, "summary1", "feedback1", model, max_tokens, base_url, headers)
    df = refineWFeedback(df, "summary1", "feedback1", "summary2_W_feedback", model, max_tokens, base_url, headers)
    return df

def summary_refinement_WO_feedback(df, model, max_tokens, base_url, headers):
    df = refineWOFeedback(df, "summary1", "summary2_WO_feedback", model, max_tokens, base_url, headers)
    return df

def main(input_path, output_path):
    api_key = os.environ.get("LEPTON_API_TOKEN", "gnULUWYeR9hzy2uygiXoqvi1gkoPuVKD")
    base_url = "https://llama3-1-8b.lepton.run/api/v1/"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(list(data.items()), columns=['id', 'reviews'])
    df = df.reset_index(drop=True)

    model = "llama3.1-8b"
    max_tokens = 2048

    df = summary_refinement_W_feedback(df, model, max_tokens, base_url, headers)
    df = summary_refinement_WO_feedback(df, model, max_tokens, base_url, headers)

    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input JSON and output CSV paths.")
    parser.add_argument('input_path', type=str, help='Path to the input JSON file')
    parser.add_argument('output_path', type=str, help='Path to the output CSV file')
    args = parser.parse_args()

    main(args.input_path, args.output_path)