import argparse
import os
import json

from dotenv import load_dotenv

from pathlib import Path


from llm_utils import load_model

from tqdm import tqdm
from datasets.permpst import load_permpst
from datasets.recipe import load_recipe
from datasets.books import load_books
import pandas as pd


def get_permpst_path() -> str:
    return "/work/gh35/h35008/preference-prediction-prompt/data/permpst/raw/review.valid.c5.jsonl"


def get_recipe_path() -> str:
    return "/work/gh35/h35008/preference-prediction-prompt/data/recipe/formatted/PP_test_5.jsonl"


def get_books_path() -> str:
    return "/work/gh35/h35008/preference-prediction-prompt/data/books/formatted/sampled_books.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")
    parser.add_argument("--dataset", type=str, default="permpst")

    load_dotenv()

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset

    if dataset_name == "permpst":
        permpst_path = get_permpst_path()
        dataset = load_permpst(permpst_path, k=5)
    elif dataset_name == "recipe":
        recipe_path = get_recipe_path()
        dataset = load_recipe(recipe_path, k=5)
    elif dataset_name == "books":
        books_path = get_books_path()
        dataset = load_books(books_path, k=5)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    job_id = os.environ["PJM_JOBID"]
    run_id = job_id

    root_path = Path(__file__).parent.parent
    output_root_path = root_path / "outputs" / run_id
    output_root_path.mkdir(exist_ok=True, parents=True)

    model = load_model(model_name)

    result_rows = []
    for instance in tqdm(dataset):
        prompt = instance.make_user_profile_prompt()
        response = model(prompt)
        result_rows.append(
            [
                json.dumps(prompt),
                response,
            ]
        )

    output_df = pd.DataFrame(
        result_rows,
        columns=[
            "prompt",
            "raw_response",
        ],
    )
    output_path = (
        Path("/work/gh35/h35008/preference-prediction-prompt/data/")
        / dataset_name
        / "user_profile"
        / model_name
        / "output.csv"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_df.to_csv(output_path, index=False)
