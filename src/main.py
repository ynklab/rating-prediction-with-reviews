import argparse
import os
import random
import json

from dotenv import load_dotenv

from pathlib import Path


from llm_utils import load_model

from tqdm import tqdm
from datasets.permpst import load_permpst
import pandas as pd

PERMPST_PATH = "/work/gh35/h35008/preference-prediction-prompt/data/permpst/raw/review.valid.c3.jsonl"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--debug", action="store_true", default=False)

    load_dotenv()

    args = parser.parse_args()
    model_name = args.model
    mode = args.mode
    debug = args.debug

    job_id = os.environ["PJM_JOBID"]
    run_id = f"{job_id}_{model_name}_{mode}"
    if debug:
        run_id = f"0_debug_{run_id}"
    print(f"Run ID: {run_id}")

    root_path = Path(__file__).parent.parent
    output_root_path = root_path / "outputs" / run_id
    output_root_path.mkdir(exist_ok=True, parents=True)

    model = load_model(model_name)

    dataset = load_permpst(PERMPST_PATH)

    if debug:
        random.seed(0)
        dataset = random.sample(dataset, 100)

    result_rows = []
    for instance in tqdm(dataset):
        preference = None
        if mode == "kar":
            preference_prompt = instance.make_reasoning_prompt(mode)
            preference = model(preference_prompt)
        prompt = instance.make_prompt(mode, preference)
        response = model(prompt)
        result_rows.append(
            [
                json.dumps(prompt),
                response,
                instance.label_review,
                instance.label_score,
                preference,
            ]
        )

    output_df = pd.DataFrame(
        result_rows,
        columns=["prompt", "raw_response", "label_review", "label_score", "preference"],
    )

    output = output_root_path / "raw_output.csv"
    output_df.to_csv(output, index=False)
