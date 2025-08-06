from typing import List, Optional

from datasets import Dataset


def prepare_gsm8k_chat_messages(question: str) -> str:
    user_message_template = "{question}\nLet's think step by step. Put your final answer at the end with 'The answer is: .'"
    user_message = user_message_template.format(question=question)
    return [
        {"role": "user", "content": user_message},
    ]


def prepare_gsm8k_batch_chat_messages(
    questions: Optional[List[str]] = None,
    dataset: Optional[Dataset] = None,
) -> List[str]:
    if questions is None and dataset is None:
        raise ValueError("Either questions or dataset must be provided.")

    if questions is not None:
        user_messages = [
            prepare_gsm8k_chat_messages(question) for question in questions
        ]
    else:
        if "question" not in dataset.column_names:
            raise ValueError("Dataset must contain a 'question' column.")

        questions = dataset["question"]
        user_messages = [
            prepare_gsm8k_chat_messages(question) for question in questions
        ]

    return user_messages


prepare_batch_chat_messages_fns = {
    "gsm8k": prepare_gsm8k_batch_chat_messages,
}

prepare_chat_messages_fns = {
    "gsm8k": prepare_gsm8k_chat_messages,
}

__all__ = ["prepare_batch_chat_messages_fns", "prepare_chat_messages_fns"]
