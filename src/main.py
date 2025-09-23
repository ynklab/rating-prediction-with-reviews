import argparse
import os
import random
import json

from dotenv import load_dotenv

from pathlib import Path


from llm_utils import load_model

from tqdm import tqdm
from datasets.permpst import load_permpst
from datasets.recipe import load_recipe
from datasets.books import load_books
import pandas as pd

this_dir = Path(__file__).parent


def get_permpst_path() -> str:
    return this_dir / "../data/permpst/raw/review.valid.c5.jsonl"


def get_recipe_path() -> str:
    return this_dir / "../data/recipe/formatted/PP_test_5.jsonl"


def get_books_path() -> str:
    return this_dir / "../data/books/formatted/sampled_books.jsonl"


def get_books_short_path() -> str:
    return this_dir / "../data/books-short/formatted/sampled_books_short_review.jsonl"


def get_books_even_distr_path() -> str:
    return (
        this_dir / "../data/books-even-distr/formatted/sampled_books_even_distr.jsonl"
    )


def get_permpst_shuffle_path() -> str:
    return this_dir / "../data/permpst-shuffle/raw/review.valid.c5.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")
    parser.add_argument("--profile-model", type=str, required=False, default=None)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--from-idx", type=int, required=False, default=None)
    parser.add_argument("--to-idx", type=int, required=False, default=None)
    parser.add_argument("--dataset", type=str, default="permpst")

    load_dotenv()

    args = parser.parse_args()
    model_name = args.model
    profile_model_name = model_name
    if args.profile_model is not None:
        profile_model_name = args.profile_model
    mode = args.mode
    debug = args.debug
    k = args.k
    dataset_name = args.dataset

    if dataset_name == "permpst":
        permpst_path = get_permpst_path()
        dataset = load_permpst(permpst_path, k=k)
    elif dataset_name == "recipe":
        recipe_path = get_recipe_path()
        dataset = load_recipe(recipe_path, k=k)
    elif dataset_name == "books":
        books_path = get_books_path()
        dataset = load_books(books_path, k=k)
    elif dataset_name == "permpst-shuffle":
        permpst_shuffle_path = get_permpst_shuffle_path()
        dataset = load_permpst(permpst_shuffle_path, k=k)
    elif dataset_name == "books-short":
        books_path = get_books_short_path()
        dataset = load_books(books_path, k=k)
    elif dataset_name == "books-even-distr":
        books_path = get_books_even_distr_path()
        dataset = load_books(books_path, k=k)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if mode in [
        "user-profile",
        "user-profile-readreview",
        "user-profile-noreview",
        "user-profile-noicl",
    ]:
        preference_csv = pd.read_csv(
            this_dir
            / f"../data/{dataset_name}/user_profile/{profile_model_name}/output.csv"
        )

        preference_list = preference_csv["raw_response"].tolist()

        for i, instance in enumerate(dataset):
            instance.preference = preference_list[i]

    from_idx = args.from_idx
    if from_idx is None:
        from_idx = 0

    to_idx = args.to_idx
    if to_idx is None:
        to_idx = len(dataset)

    job_id = os.environ["PJM_JOBID"]
    run_id = f"{job_id}_{model_name}_{mode}_{dataset_name}_{k}_from_{from_idx}_to_{to_idx}_profile_{profile_model_name}"
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
        score_trend = None
        recommendation = None
        if mode == "kar" or mode == "kar-llmrec":
            preference_prompt = instance.make_reasoning_prompt(mode)
            preference = model(preference_prompt)
        if mode == "kar-llmrec":
            recommendation_prompt = instance.make_recommendation_prompt(mode)
            recommendation = model(recommendation_prompt)
        if mode == "scoresumm":
            score_trend_prompt = instance.make_score_trend_prompt(mode)
            score_trend = model(score_trend_prompt)
        if mode in [
            "user-profile",
            "user-profile-readreview",
            "user-profile-noreview",
            "user-profile-noicl",
        ]:
            preference = instance.preference

        prompt = instance.make_prompt(
            mode, preference, score_trend=score_trend, recommendation=recommendation
        )
        response = model(prompt)
        result_rows.append(
            [
                json.dumps(prompt),
                response,
                instance.label_review,
                instance.label_score,
                preference,
                score_trend,
                recommendation,
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
            "recommendation",
        ],
    )

    output = output_root_path / "raw_output.csv"
    output_df.to_csv(output, index=False)
