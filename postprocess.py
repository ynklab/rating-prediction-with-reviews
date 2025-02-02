import re
import json
import argparse
import pandas as pd
from pathlib import Path

import scipy
import sklearn


def extract_json_from_output(text):
    """
    Extracts and parses the JSON portion from the given text.

    First, it attempts to locate a JSON block enclosed in ```json ... ```.
    If no such block is found, it tries to parse the entire text as JSON.

    Returns:
        A Python dictionary if JSON is successfully parsed, otherwise None.
    """
    # Regular expression pattern to capture JSON content inside ```json ... ```
    pattern = r"```json\s*(\{.*?\})\s*```"
    try:
        match = re.search(pattern, text, re.DOTALL)
    except Exception as e:
        print("Failed to search for JSON block:", e)
        print("Original text: ", text)
        return {}

    if match:
        json_str = match.group(1)
    else:
        # No triple-backtick block found; attempt to extract JSON by finding balanced braces
        start = text.find("{")
        if start == -1:
            print("No opening brace found.")
            return None

        # Use a counter to find the matching closing brace
        counter = 0
        end = None
        for i, char in enumerate(text[start:], start):
            if char == "{":
                counter += 1
            elif char == "}":
                counter -= 1
                if counter == 0:
                    end = i
                    break

        if end is None:
            print("Could not find a matching closing brace for the JSON object.")
            return None

        json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        print("Original text: ", text)
        return {}


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str)

    args = parser.parse_args()

    output_path = Path(args.output_path)

    output_df = pd.read_csv(output_path / "raw_output.csv")

    output_dicts = [
        extract_json_from_output(response) for response in output_df["raw_response"]
    ]

    output_df["output_review"] = [d.get("Review") for d in output_dicts]
    output_df["output_score"] = [d.get("Score") for d in output_dicts]

    output_df.to_csv(output_path / "processed_output.csv", index=False)

    # Measure the performance
    # Ratio of None score
    none_score_ratio = output_df["output_score"].isnull().mean()

    # Accuracy
    accuracy = (output_df["output_score"] == output_df["label_score"]).mean()

    non_null_output_df = output_df.dropna(subset=["output_score"])

    non_null_label = non_null_output_df["label_score"]
    non_null_output = non_null_output_df["output_score"]

    # Kendall correlation
    kendall_corr, _ = scipy.stats.kendalltau(non_null_label, non_null_output)

    # Spearman correlation
    spearman_corr, _ = scipy.stats.spearmanr(non_null_label, non_null_output)

    # MAE
    mae = sklearn.metrics.mean_absolute_error(non_null_label, non_null_output)

    # MSE
    mse = sklearn.metrics.mean_squared_error(non_null_label, non_null_output)

    output_stat_path = output_path / "output_stats.json"
    with open(output_stat_path, "w") as f:
        json.dump(
            {
                "none_score_ratio": none_score_ratio,
                "accuracy": accuracy,
                "kendall_corr": kendall_corr,
                "spearman_corr": spearman_corr,
                "mae": mae,
                "mse": mse,
            },
            f,
            indent=4,
        )
