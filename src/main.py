import argparse
import os
import random
import json

from dotenv import load_dotenv

from pathlib import Path


from llm_utils import load_model

from tqdm import tqdm
from datasets.permpst import load_permpst
from datasets.openai_tldr_axis import load_openai_tldr_axis
from datasets.recipe import load_recipe
from datasets.books import load_books
import pandas as pd


def get_permpst_path(k: int) -> str:
    return f"/work/gh35/h35008/preference-prediction-prompt/data/permpst/raw/review.valid.c{k}.jsonl"


def get_tldr_path(k: int) -> str:
    # TODO: Use k
    return "/work/gh35/h35008/preference-prediction-prompt/data/openai-tldr-axis/raw/valid.c3.jsonl"


def get_recipe_path(k: int) -> str:
    return f"/work/gh35/h35008/preference-prediction-prompt/data/recipe/formatted/PP_test_{k}.jsonl"


def get_books_path(k: int) -> str:
    # TODO: Use k?
    assert k == 5
    return "/work/gh35/h35008/preference-prediction-prompt/data/books/formatted/sampled_books.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--from-idx", type=int, required=False, default=None)
    parser.add_argument("--to-idx", type=int, required=False, default=None)
    parser.add_argument("--dataset", type=str, default="permpst")

    load_dotenv()

    args = parser.parse_args()
    model_name = args.model
    mode = args.mode
    debug = args.debug
    k = args.k
    dataset_name = args.dataset

    if dataset_name == "permpst":
        permpst_path = get_permpst_path(k)
        dataset = load_permpst(permpst_path)
    elif dataset_name == "tldr":
        tldr_path = get_tldr_path(k)
        dataset = load_openai_tldr_axis(tldr_path)
    elif dataset_name == "recipe":
        recipe_path = get_recipe_path(k)
        dataset = load_recipe(recipe_path)
    elif dataset_name == "books":
        books_path = get_books_path(k)
        dataset = load_books(books_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    from_idx = args.from_idx
    if from_idx is None:
        from_idx = 0

    to_idx = args.to_idx
    if to_idx is None:
        to_idx = len(dataset)

    job_id = os.environ["PJM_JOBID"]
    run_id = (
        f"{job_id}_{model_name}_{mode}_{dataset_name}_{k}_from_{from_idx}_to_{to_idx}"
    )
    if debug:
        run_id = f"0_debug_{run_id}"
    print(f"Run ID: {run_id}")

    dataset = dataset[from_idx:to_idx]

    root_path = Path(__file__).parent.parent
    output_root_path = root_path / "outputs" / run_id
    output_root_path.mkdir(exist_ok=True, parents=True)

    model = load_model(model_name)

    if debug:
        random.seed(0)
        dataset = random.sample(dataset, min(100, len(dataset)))

    result_rows = []
    for instance in tqdm(dataset):
        preference = None
        if mode == "kar":
            preference_prompt = instance.make_reasoning_prompt(mode)
            preference = model(preference_prompt)
        elif mode == "scoresumm":
            score_trend_prompt = instance.make_score_trend_prompt(mode)
            score_trend = model(score_trend_prompt)

        prompt = instance.make_prompt(mode, preference, score_trend=score_trend)
        response = model(prompt)
        result_rows.append(
            [
                json.dumps(prompt),
                response,
                instance.label_review,
                instance.label_score,
                preference,
                score_trend,
            ]
        )

    output_df = pd.DataFrame(
        result_rows,
        columns=[
            "prompt",
            "raw_response",
            "label_review",
            "label_score",
            "preference",
            "score_trend",
        ],
    )

    output = output_root_path / "raw_output.csv"
    output_df.to_csv(output, index=False)
