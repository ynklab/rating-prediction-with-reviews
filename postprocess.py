import re
import json
import argparse
import pandas as pd
from pathlib import Path
import numpy as np

import scipy
import sklearn


def extract_reviewer_scores_from_prompt(prompt_json_str) -> list[int]:
    """
    Extracts the reviewer scores from the given text.

    Returns:
        A list of reviewer scores if found, otherwise an empty list.
    """
    prompt_json = json.loads(prompt_json_str)
    user_prompt = prompt_json[1]["content"]
    # Extract the patterns of "Score": <score: int>
    pattern = r'"Score": \s*(\d+)'
    scores = re.findall(pattern, user_prompt)
    return [int(score) for score in scores]


def fix_json_string(json_str):
    """
    Fixes the JSON string by escaping unescaped double quotes within the Review value.

    Assumes that the JSON should only contain the keys "Review" and "Score".
    """
    # Locate the "Review" key in the JSON string.
    review_key_pattern = r'("Review"\s*:\s*")'
    match = re.search(review_key_pattern, json_str)
    if not match:
        # If no "Review" key is found, return the string as-is.
        return json_str

    # Determine the start index of the Review value.
    start_index = match.end()

    # Find the ending quote of the Review value.
    # We assume that either the next key ("Score") or the end of the JSON object marks the end.
    score_index = json_str.find('"Score"', start_index)
    if score_index != -1:
        # Locate the last quote before the "Score" key.
        end_index = json_str.rfind('"', start_index, score_index)
    else:
        # Fallback: find the last quote in the string (should be before the closing brace).
        end_index = json_str.rfind('"')

    if end_index == -1:
        return json_str

    # Extract the Review value.
    review_value = json_str[start_index:end_index]

    # Escape any unescaped double quotes within the review value.
    # The regex replaces any " not preceded by a backslash.
    fixed_value = re.sub(r'(?<!\\)"', r'\\"', review_value)

    # Reconstruct the JSON string with the fixed Review value.
    fixed_json_str = json_str[:start_index] + fixed_value + json_str[end_index:]
    return fixed_json_str


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
            return {}

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
            print(text)
            return {}

        json_str = text[start : end + 1]

    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print("Initial JSON decoding failed:", e)
        print("Original text:", json_str)
        # Try to fix the JSON string (for instance, escaping unescaped quotes in the Review field)
        fixed_json_str = fix_json_string(json_str)
        try:
            data = json.loads(fixed_json_str)
            # Filter out any keys other than "Review" and "Score"
            data = {k: v for k, v in data.items() if k in ["Review", "Score"]}
            return data
        except json.JSONDecodeError as e2:
            print("Failed to decode JSON after fix:", e2)
            print("Fixed text:", fixed_json_str)
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

    reviewer_scores_jsons = [
        json.dumps(extract_reviewer_scores_from_prompt(prompt))
        for prompt in output_df["prompt"]
    ]

    output_df["output_review"] = [d.get("Review") for d in output_dicts]
    output_df["output_score"] = [d.get("Score") for d in output_dicts]
    output_df["reviewer_scores"] = reviewer_scores_jsons

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

    reviewer_score_avgs = [
        np.mean(json.loads(scores)) for scores in non_null_output_df["reviewer_scores"]
    ]

    # Repeat using reviewer avgs as the output
    reviewer_avg_kendall_corr, _ = scipy.stats.kendalltau(
        non_null_label, reviewer_score_avgs
    )
    reviewer_avg_spearman_corr, _ = scipy.stats.spearmanr(
        non_null_label, reviewer_score_avgs
    )
    reviewer_avg_mae = sklearn.metrics.mean_absolute_error(
        non_null_label, reviewer_score_avgs
    )
    reviewer_avg_mse = sklearn.metrics.mean_squared_error(
        non_null_label, reviewer_score_avgs
    )

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
                "reviewer_avg_kendall_corr": reviewer_avg_kendall_corr,
                "reviewer_avg_spearman_corr": reviewer_avg_spearman_corr,
                "reviewer_avg_mae": reviewer_avg_mae,
                "reviewer_avg_mse": reviewer_avg_mse,
            },
            f,
            indent=4,
        )
