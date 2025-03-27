from pathlib import Path
import pandas as pd

ORIGINAL_SCORE_SYSTEM_PROMPT = "You function as an insightful assistant whose role is to assist individuals in making decisions that align with their personal preferences. Use your understanding of their likes, dislikes, and inclinations to provide relevant and thoughtful recommendations."
ORIGINAL_SCORE_CASE_TEMPLATE = """[The Start of Recipe {n}]
{recipe}
[The End of Recipe {n}]
[Review]
```json
{{
  "Review": "{review}",
  "Score": {score}
}}
```
"""

ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several recipes, each accompanied by a review from the same user. Your task is to analyze both the recipe and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new recipe and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

Please follow the above user and give a review for the given recipe. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-5, 1 is the lowest and 5 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Recipe]
{recipe}
[The End of Recipe]
"""

ORIGINAL_ASSISTANT_BEGIN = "[Review] Here is the Json format of the review: "

NOREVIEWS_SCORE_CASE_TEMPLATE = """[The Start of Recipe {n}]
{recipe}
[The End of Recipe {n}]
[Review]
```json
{{
  "Score": {score}
}}
```
"""

NOREVIEW_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several recipes, each accompanied by a review from the same user. Your task is to analyze both the recipe and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new recipe and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

Please follow the above user and give a review for the given recipe. Your response should strictly follow the format: 
```json
{{
  "Score": <1-5, 1 is the lowest and 5 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Recipe]
{recipe}
[The End of Recipe]
"""
COT_ASSISTANT_BEGIN = "Let's think step by step."
PS_ASSISTANT_BEGIN = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."


KAR_PREFERENCE_REASONING_TEMPLATE = """A user's past recipe reviews are listed below:

{icl_example}

Analyze the user's preferences. Provide clear explanations based on details from thep past reviews and other pertinent factors.
"""

KAR_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several recipes, each accompanied by a review from the same user. Your task is to analyze both the recipe and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new recipe and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

The preference of him/her is analyzed as follows:

{preference}

Please follow the above user and give a review for the given recipe. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-5, 1 is the lowest and 5 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Recipe]
{recipe}
[The End of Recipe]
"""


class RecipeInstance:
    def __init__(self, instance: dict):
        self.examples = instance["demonstrations"]
        self.target = instance["target"]
        self.label_review = self.target["review"]
        self.label_score = int(self.target["score"])

    def make_reasoning_prompt(self, mode: str):
        if mode not in ["kar"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                n=i + 1,
                recipe=x["recipe_text"],
                review=x["review"],
                score=int(x["score"]),
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
        if mode not in ["original", "noreview", "readreview", "cot", "ps", "kar"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            if mode == "noreview":
                case = NOREVIEWS_SCORE_CASE_TEMPLATE.format(
                    n=i + 1,
                    recipe=x["recipe_text"],
                    score=int(x["score"]),
                )
            else:
                case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                    n=i + 1,
                    recipe=x["recipe_text"],
                    review=x["review"],
                    score=int(x["score"]),
                )
            icl_content += case + "\n"

        if mode == "noreview" or mode == "readreview":
            prompt = NOREVIEW_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                recipe=self.target["recipe_text"],
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "original":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                recipe=self.target["recipe_text"],
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "cot":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                recipe=self.target["recipe_text"],
            )
            assistant_begin = COT_ASSISTANT_BEGIN
        elif mode == "ps":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                recipe=self.target["recipe_text"],
            )
            assistant_begin = PS_ASSISTANT_BEGIN
        else:
            prompt = KAR_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                recipe=self.target["recipe_text"],
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


def load_recipe(dataset: Path) -> list[RecipeInstance]:
    return [
        RecipeInstance(instance)
        for instance in pd.read_json(dataset, lines=True).to_dict(orient="records")
    ]
