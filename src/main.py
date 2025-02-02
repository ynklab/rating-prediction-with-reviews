import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-31-8b-i")

    load_dotenv()

    model = parser.parse_args().model
    print(f"Hello, {model}!")
