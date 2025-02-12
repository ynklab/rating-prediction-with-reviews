from pathlib import Path
import pandas as pd

ORIGINAL_SCORE_SYSTEM_PROMPT = "You function as an insightful assistant whose role is to assist individuals in making decisions that align with their personal preferences. Use your understanding of their likes, dislikes, and inclinations to provide relevant and thoughtful recommendations."
ORIGINAL_SCORE_CASE_TEMPLATE = """[The Start of Text {n}]
{text}
[The End of Text {n}]
[The Start of Summary {n}]
{summary}
[The End of Summary {n}]
[Evaluation]
```json
{{
  "Note": "{note}",
  "Score": {score}
}}
```
"""

ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several pairs of a text and a summary of it. Your task is to analyze both the summaries and the corresponding evaluations to discern the evaluator's preferences. Afterward, consider a new pair of a text and a summary, and create an evaluation that you believe this evaluator would write based on the established preferences. 

{icl_example}

Please follow the above critic and give an evaluation for the given summary. Your response should strictly follow the format: 
```json
{{
  "Note": "<proposed note conforms to style demonstrated in the previous evaluations>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Text]
{text}
[The End of Text]
[The Start of Summary]
{summary}
[The End of Summary]
"""

ORIGINAL_ASSISTANT_BEGIN = "[Evaluation] Here is the Json format of the evaluation: "
COT_ASSISTANT_BEGIN = "Let's think step by step."
PS_ASSISTANT_BEGIN = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."


KAR_PREFERENCE_REASONING_TEMPLATE = """An evaluator's past evaluations of text summary quality are listed below:

{icl_example}

Analyze the evaluator's preferences. Provide clear explanations based on details from their past evaluations and other pertinent factors.
"""


KAR_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several pairs of a text and a summary of it. Your task is to analyze both the summaries and the corresponding evaluations to discern the evaluator's preferences. Afterward, consider a new pair of a text and a summary, and create a evaluation that you believe this evaluator would write based on the established preferences. 

{icl_example}

The preference of him/her is analyzed as follows:

{preference}

Please follow the above critic and give an evaluation for the given summary. Your response should strictly follow the format: 
```json
{{
  "Note": "<proposed note conforms to style demonstrated in the previous evaluations>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Text]
{text}
[The End of Text]
[The Start of Summary]
{summary}
[The End of Summary]
"""


class OpenAITLDRAxisInstance:
    def __init__(self, instance: dict):
        self.examples = instance["demonstrations"]
        self.target = instance["target"]
        self.label_review = self.target["summary"]["note"]
        self.label_score = self.target["summary"]["axes"]["overall"]

    def make_reasoning_prompt(self, mode: str):
        if mode not in ["kar"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                n=i + 1,
                text=x["info"]["post"],
                summary=x["summary"]["text"],
                note=x["summary"]["note"],
                score=x["summary"]["axes"]["overall"],
            )
            icl_content += case + "\n"

        prompt = KAR_PREFERENCE_REASONING_TEMPLATE.format(
            icl_example=icl_content,
        )
        messages = [
            {"role": "system", "content": ORIGINAL_SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        return messages

    def make_prompt(self, mode: str, preference: str | None = None):
        if mode not in ["original", "cot", "ps", "kar"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                n=i + 1,
                text=x["info"]["post"],
                summary=x["summary"]["text"],
                note=x["summary"]["note"],
                score=x["summary"]["axes"]["overall"],
            )
            icl_content += case + "\n"

        if mode == "original":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                text=self.target["info"]["post"],
                summary=self.target["summary"]["text"],
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "cot":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                text=self.target["info"]["post"],
                summary=self.target["summary"]["text"],
            )
            assistant_begin = COT_ASSISTANT_BEGIN
        elif mode == "ps":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                text=self.target["info"]["post"],
                summary=self.target["summary"]["text"],
            )
            assistant_begin = PS_ASSISTANT_BEGIN
        else:
            prompt = KAR_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                text=self.target["info"]["post"],
                summary=self.target["summary"]["text"],
                preference=preference,
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        messages = [
            {"role": "system", "content": ORIGINAL_SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": assistant_begin,
            },
        ]

        return messages


def load_openai_tldr_axis(dataset: Path) -> list[OpenAITLDRAxisInstance]:
    return [
        OpenAITLDRAxisInstance(instance)
        for instance in pd.read_json(dataset, lines=True).to_dict(orient="records")
    ]
