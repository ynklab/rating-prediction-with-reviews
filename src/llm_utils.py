import torch
import transformers
import hashlib
import numpy as np
import os

from huggingface_hub import login


def get_model_id(model: str):
    if model == "llama-31-8b-i":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif model == "llama-31-8b":
        return "meta-llama/Llama-3.1-8B"
    elif model == "r1-distill-llama":
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        raise ValueError(f"Invalid model: {model}")


def fix_torch_seed(seed: str):
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    torch.manual_seed(int(digest, 16) % (2**32))


def fix_numpy_seed(seed: str):
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    np.random.seed(int(digest, 16) % (2**32))


class LlamaPipeline:
    def __init__(self, model_id: str):
        self.model = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

    def build_prompt(self, prompt: list[dict]) -> str:
        prompt_str = "<|begin_of_text|>"
        for i, p in enumerate(prompt):
            if p["role"] == "system":
                prompt_str += "<|start_header_id|>system<|end_header_id|>"
                prompt_str += p["content"]
                prompt_str += "<|eot_id|>"
            elif p["role"] == "user":
                prompt_str += "<|start_header_id|>user<|end_header_id|>"
                prompt_str += p["content"]
                prompt_str += "<|eot_id|>"
            elif p["role"] == "assistant":
                prompt_str += "<|start_header_id|>assistant<|end_header_id|>"
                prompt_str += p["content"]
                if i != len(prompt) - 1:
                    prompt_str += "<|eot_id|>"
            else:
                raise ValueError(f"Invalid role: {p['role']}")
        if prompt[-1]["role"] != "assistant":
            prompt_str += "<|start_header_id|>assistant<|end_header_id|>"

        return prompt_str

    def __call__(self, prompt: list[dict]) -> str:
        prompt_str = self.build_prompt(prompt)

        # Generate the response from the model
        response = self.model(
            prompt_str,
            max_new_tokens=768,
            return_full_text=False,
            temperature=0.01,
            pad_token_id=self.model.tokenizer.eos_token_id,  # type: ignore
        )

        return response[0]["generated_text"]


MODEL = None


def load_model(model: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    else:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)

        model_id = get_model_id(model)
        MODEL = LlamaPipeline(model_id)
        return MODEL
