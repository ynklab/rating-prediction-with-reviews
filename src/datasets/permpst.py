from pathlib import Path
import pandas as pd

ORIGINAL_SCORE_SYSTEM_PROMPT = "You function as an insightful assistant whose role is to assist individuals in making decisions that align with their personal preferences. Use your understanding of their likes, dislikes, and inclinations to provide relevant and thoughtful recommendations."
ORIGINAL_SCORE_CASE_TEMPLATE = """[The Start of Plot {n}]
{plot}
[The End of Plot {n}]
[Review]
```json
{{
  "Review": "{review}",
  "Score": {score}
}}
```
"""

ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several plot summaries, each accompanied by a review from the same critic. Your task is to analyze both the plot summaries and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new plot and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

Please follow the above critic and give a review for the given plot. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Plot]
{plot}
[The End of Plot]
"""

ORIGINAL_ASSISTANT_BEGIN = "[Review] Here is the Json format of the review: "

NOREVIEW_SCORE_CASE_TEMPLATE = """[The Start of Plot {n}]
{plot}
[The End of Plot {n}]
[Review]
```json
{{
  "Score": {score}
}}
```
"""

NOREVIEW_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several plot summaries, each accompanied by a review from the same critic. Your task is to analyze both the plot summaries and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new plot and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

Please follow the above critic and give a review for the given plot. Your response should strictly follow the format: 
```json
{{
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Plot]
{plot}
[The End of Plot]
"""

COT_ASSISTANT_BEGIN = "Let's think step by step."
PS_ASSISTANT_BEGIN = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step."


KAR_PREFERENCE_REASONING_TEMPLATE = """A critic's past movie reviews are listed below:

{icl_example}

Analyze the critic's preferences. Provide clear explanations based on details from thep past reviews and other pertinent factors.
"""

KAR_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several plot summaries, each accompanied by a review from the same critic. Your task is to analyze both the plot summaries and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new plot and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

The preference of him/her is analyzed as follows:

{preference}

Please follow the above critic and give a review for the given plot. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Plot]
{plot}
[The End of Plot]
"""

SCORESUMM_REASONING_TEMPLATE = """A critic's past movie reviews are listed below:

{icl_example}

Based on this user’s past reviews, what are the most common scores they give for positive and negative reviews?
Answer in the following form:
most common positive score: <most common positive score>, most common negative score: <most common negative score>
"""

SCORESUMM_SCORE_PROBLEM_PROMPT_TEMPLATE = """[User Question] You will be presented with several plot summaries, each accompanied by a review from the same critic. Your task is to analyze both the plot summaries and the corresponding reviews to discern the reviewer's preferences. Afterward, consider a new plot and create a review that you believe this reviewer would write based on the established preferences. 

{icl_example}

The trend of review scores given by this user is analyzed as follows:
{score_trend}

Please follow the above critic and give a review for the given plot. Your response should strictly follow the format: 
```json
{{
  "Review": "<proposed review conforms to style demonstrated in the previous reviews>",
  "Score": <1-10, 1 is the lowest and 10 is the highest>
}}
```
Please remember to replace the placeholder text within the "<>" with the appropriate details of your response.

[The Start of Plot]
{plot}
[The End of Plot]
"""


class PerMPSTInstance:
    def __init__(self, instance: dict):
        self.examples = instance["examples"][:-1]
        self.target = instance["examples"][-1]
        self.label_review = self.target["clean_review"]
        self.label_score = self.target["score"]

    def make_score_trend_prompt(self, mode: str):
        if mode not in ["scoresumm"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                n=i + 1,
                plot=x["summ_plot"],
                review=x["clean_review"],
                score=x["score"],
            )
            icl_content += case + "\n"

        prompt = SCORESUMM_REASONING_TEMPLATE.format(
            icl_example=icl_content,
        )
        messages = [
            {"role": "system", "content": ORIGINAL_SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        return messages

    def make_reasoning_prompt(self, mode: str):
        if mode not in ["kar"]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                n=i + 1,
                plot=x["summ_plot"],
                review=x["clean_review"],
                score=x["score"],
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

    def make_prompt(
        self, mode: str, preference: str | None = None, score_trend: str | None = None
    ):
        if mode not in [
            "original",
            "noreview",
            "readreview",
            "cot",
            "ps",
            "kar",
            "scoresumm",
        ]:
            raise ValueError(f"Invalid mode: {mode}")
        icl_content = ""
        for i, x in enumerate(self.examples):
            if mode == "noreview":
                case = NOREVIEW_SCORE_CASE_TEMPLATE.format(
                    n=i + 1,
                    plot=x["summ_plot"],
                    score=x["score"],
                )
            else:
                case = ORIGINAL_SCORE_CASE_TEMPLATE.format(
                    n=i + 1,
                    plot=x["summ_plot"],
                    review=x["clean_review"],
                    score=x["score"],
                )
            icl_content += case + "\n"

        if mode == "noreview" or mode == "readreview":
            prompt = NOREVIEW_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "original":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "cot":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
            )
            assistant_begin = COT_ASSISTANT_BEGIN
        elif mode == "ps":
            prompt = ORIGINAL_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
            )
            assistant_begin = PS_ASSISTANT_BEGIN
        elif mode == "kar":
            prompt = KAR_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
                preference=preference,
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        elif mode == "scoresumm":
            prompt = SCORESUMM_SCORE_PROBLEM_PROMPT_TEMPLATE.format(
                icl_example=icl_content,
                plot=self.target["summ_plot"],
                score_trend=preference,
            )
            assistant_begin = ORIGINAL_ASSISTANT_BEGIN
        else:
            raise ValueError(f"Invalid mode: {mode}")
        messages = [
            {"role": "system", "content": ORIGINAL_SCORE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {
                "role": "assistant",
                "content": assistant_begin,
            },
        ]

        return messages


def load_permpst(dataset: Path) -> list[PerMPSTInstance]:
    return [
        PerMPSTInstance(instance)
        for instance in pd.read_json(dataset, lines=True).to_dict(orient="records")
    ]
