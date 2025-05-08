from typing import List, Union

from datasets import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


# TODO: Name this better
def prepare_messages_gsm8k(question: str) -> str:
    user_message_template = "{question}\nLet's think step by step. Put your final answer at the end with 'The answer is: .'"
    user_message = user_message_template.format(question=question)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]


def prepare_prompts_gsm8k(
    dataset: Dataset,
    tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
) -> List[str]:
    if "question" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'question' column.")

    questions = dataset["question"]
    user_message_template = "{question}\nLet's think step by step. Put your final answer at the end with 'The answer is: .'"
    user_messages = [
        user_message_template.format(question=question) for question in questions
    ]
    # Make messages
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message},
        ]
        for user_message in user_messages
    ]
    # Make prompts
    prompts = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # List[str]

    return prompts


prepare_prompts_fns = {
    "gsm8k": prepare_prompts_gsm8k,
}

__all__ = ["prepare_prompts_fns"]
