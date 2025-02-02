import argparse
import os

from dotenv import load_dotenv

from pathlib import Path


from llm_utils import load_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")
    parser.add_argument("--debug", action="store_true", default=False)

    load_dotenv()

    args = parser.parse_args()
    model_name = args.model
    debug = args.debug

    job_id = os.environ["PJM_JOBID"]
    run_id = f"{job_id}_{model_name}"
    if debug:
        run_id = f"0_debug_{run_id}"
    print(f"Run ID: {run_id}")

    root_path = Path(__file__).parent.parent
    output_root_path = root_path / "outputs" / run_id
    output_root_path.mkdir(exist_ok=True, parents=True)

    model = load_model(model_name)
    prompt = [
        {"role": "user", "content": "Hello, how are you doing today?"},
    ]
    print(model(prompt))
