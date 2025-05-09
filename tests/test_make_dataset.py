from unittest.mock import patch

import pytest
from datasets import Dataset

from revise import GSM8KEvaluator
from revise.make_dataset import generate_and_evaluate, make_dataset
from revise.prompts import prepare_batch_chat_messages_fns, prepare_chat_messages_fns


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<|reserved_special_token_0|>"

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages


@pytest.mark.unit
class TestMakeDataset:
    @patch("ray.get")
    @patch("revise.generators.vllm_generator._generate_on_gpu_chat")
    @patch("transformers.AutoTokenizer")
    def test_generate_and_evaluate(
        self, mock_auto_tokenizer, mock_generate_on_gpu, mock_ray_get, mock_vllm
    ):
        mock_auto_tokenizer.return_value = MockTokenizer()
        mock_generate_on_gpu.remote.return_value = "future"
        mock_ray_get.return_value = [
            [(0, ["generated text 1"]), (1, ["generated text 2"])]
        ]

        dataset = generate_and_evaluate(
            model_path="test-model",
            dataset=Dataset.from_list(
                [
                    {"question": "question 1", "answer": "answer 1"},
                    {"question": "question 2", "answer": "answer 2"},
                ]
            ),
            evaluator=GSM8KEvaluator(),
            prepare_batch_chat_messages_fn=prepare_batch_chat_messages_fns["gsm8k"],
        )

        assert len(dataset) == 2

    def test_make_dataset(self):
        question_key = "question"
        answer_key = "answer"
        prediction_key = "prediction"
        prepare_chat_messages_fn = prepare_chat_messages_fns["gsm8k"]

        dataset = make_dataset(
            dataset=Dataset.from_list(
                [
                    {
                        question_key: "question 1",
                        answer_key: "answer 1",
                        prediction_key: "prediction 1",
                        "is_correct": True,
                    },
                    {
                        question_key: "question 2",
                        answer_key: "answer 2",
                        prediction_key: "prediction 2",
                        "is_correct": False,
                    },
                ],
            ),
            question_key=question_key,
            answer_key=answer_key,
            prediction_key=prediction_key,
            prepare_chat_messages_fn=prepare_chat_messages_fn,
        )

        assert len(dataset) == 2

        assert dataset[0][question_key] == "question 1"
        assert dataset[0][answer_key] == "answer 1"
        assert dataset[0][prediction_key] == "prediction 1"
        assert dataset[0]["is_correct"] is True

        assert dataset[1][question_key] == "question 2"
        assert dataset[1][answer_key] == "answer 2"
        assert dataset[1][prediction_key] == "prediction 2"
        assert dataset[1]["is_correct"] is False
