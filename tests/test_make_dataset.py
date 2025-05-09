from unittest.mock import patch

import pytest
from datasets import Dataset

from revise import GSM8KEvaluator
from revise.make_dataset import generate_and_evaluate
from revise.prompts import prepare_prompts_gsm8k


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<|reserved_special_token_0|>"

    def apply_chat_template(self, messages, add_generation_prompt=True):
        return messages


@pytest.mark.unit
class TestMakeDataset:
    @patch("ray.get")
    @patch("revise.generators.vllm_generator._generate_on_gpu")
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
            model_path="meta-llama/llama-3.2-1b-instruct",
            dataset=Dataset.from_list(
                [
                    {"question": "question 1", "answer": "answer 1"},
                    {"question": "question 2", "answer": "answer 2"},
                ]
            ),
            evaluator=GSM8KEvaluator(),
            prepare_batch_user_messages_fn=prepare_prompts_gsm8k,
        )

        assert len(dataset) == 2
