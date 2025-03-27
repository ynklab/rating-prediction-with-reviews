import torch
import transformers
import hashlib
import numpy as np
import os

from huggingface_hub import login
from transformers import AutoTokenizer, Gemma3ForCausalLM, AutoModelForCausalLM


def get_model_id(model: str):
    if model == "llama-31-8b-i":
        return "meta-llama/Llama-3.1-8B-Instruct"
    elif model == "llama-33-70b-i":
        return "meta-llama/Llama-3.3-70B-Instruct"
    elif model == "gemma3-12b-it":
        return "google/gemma-3-12b-it"
    elif model == "gemma3-27b-it":
        return "google/gemma-3-27b-it"
    elif model == "qwq-32b":
        return "Qwen/QwQ-32B"
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


class GemmaPipeline:
    def __init__(self, model_id: str):
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    def build_prompt(self, prompt: list[dict]) -> list[dict]:
        return [
            {
                "role": message["role"],
                "content": [
                    {
                        "type": "text",
                        "text": message["content"],
                    }
                ],
            }
            for message in prompt
        ]

    def __call__(self, prompt: list[dict]) -> str:
        # Generate the response from the model
        messages = self.build_prompt(prompt)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="longest",
            tokenizer_kwargs={
                "pad_to_multiple_of": 8,
            },
        ).to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]
        generation = self.model.generate(
            **inputs,
            max_new_tokens=768,
        )
        generation = generation[0][input_len:]
        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded


class QwQPipeline:
    def __init__(self, model_id: str):
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    def __call__(self, prompt: list[dict]) -> str:
        # Generate the response from the model
        inputs = self.processor.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.processor([inputs], return_tensors="pt").to(
            self.model.device
        )

        generated_ids = self.model.generate(**model_inputs, max_new_tokens=32768)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]

        return response


MODEL = None


def load_model(model: str):
    global MODEL
    if MODEL is not None:
        return MODEL
    else:
        hf_token = os.environ["HF_TOKEN"]
        login(token=hf_token)
        if model.startswith("gemma"):
            model_id = get_model_id(model)
            MODEL = GemmaPipeline(model_id)
            return MODEL
        elif model.startswith("qwq"):
            model_id = get_model_id(model)
            MODEL = QwQPipeline(model_id)
            return MODEL
        else:
            model_id = get_model_id(model)
            MODEL = LlamaPipeline(model_id)
            return MODEL
