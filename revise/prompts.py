from typing import List

from datasets import Dataset


# TODO: Name this better
def prepare_user_messages_gsm8k(question: str) -> str:
    user_message_template = "{question}\nLet's think step by step. Put your final answer at the end with 'The answer is: .'"
    user_message = user_message_template.format(question=question)
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]


# TODO: Name this better
def prepare_batch_user_messages_gsm8k(
    dataset: Dataset,
) -> List[str]:
    if "question" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'question' column.")

    questions = dataset["question"]
    user_messages = [prepare_user_messages_gsm8k(question) for question in questions]

    return user_messages


prepare_batch_user_messages_fns = {
    "gsm8k": prepare_batch_user_messages_gsm8k,
}

__all__ = ["prepare_batch_user_messages_fns"]
