import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from revise.args.sft import ModelArguments, SFTArguments, TrainingArguments
from revise.sft import get_dataset, get_model_and_tokenizer, main


@pytest.fixture
def model_args():
    return ModelArguments(
        model_name_or_path="meta-llama/llama-3.2-1b",
        tokenizer_name_or_path="meta-llama/llama-3.2-1b",
        device_map="cpu",
        dtype="float32",
    )


@pytest.fixture
def training_args():
    return TrainingArguments(
        output_dir=tempfile.mkdtemp(),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_steps=1,
        report_to="none",
        save_strategy="steps",
        save_steps=1,
        eval_strategy="steps",
        eval_steps=1,
    )


@pytest.fixture
def sft_args(model_args, training_args):
    return SFTArguments(model_args=model_args, training_args=training_args)


@pytest.mark.unit
class TestSFT:
    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_get_model_and_tokenizer(self, mock_tokenizer, mock_model, model_args):
        # Mock model and tokenizer
        mock_model.return_value = MagicMock(spec=AutoModelForCausalLM)
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"

        # Test model and tokenizer loading
        model, tokenizer = get_model_and_tokenizer(model_args)

        # Verify model loading
        mock_model.assert_called_once_with(
            model_args.model_name_or_path,
            device_map="cpu",
            use_cache=False,
            torch_dtype=torch.float32,
            attn_implementation=None,
        )

        # Verify tokenizer loading
        mock_tokenizer.assert_called_once_with(model_args.model_name_or_path)
        assert tokenizer.pad_token == "<|endoftext|>"

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_get_model_and_tokenizer_with_lora(self, mock_tokenizer, mock_model):
        # Test with LoRA configuration
        model_args = ModelArguments(
            model_name_or_path="meta-llama/llama-3.2-1b",
            tokenizer_name_or_path="meta-llama/llama-3.2-1b",
            device_map="cpu",
            dtype="float32",
            use_lora=True,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
        )

        mock_model.return_value = MagicMock(spec=AutoModelForCausalLM)
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"

        model, tokenizer = get_model_and_tokenizer(model_args)

        # Verify model loading with LoRA
        mock_model.assert_called_once_with(
            model_args.model_name_or_path,
            device_map="cpu",
            use_cache=False,
            torch_dtype=torch.float32,
            attn_implementation=None,
        )

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_get_model_and_tokenizer_device_map(self, mock_tokenizer, mock_model):
        # Test different device maps
        device_maps = ["cuda", "mps", "cpu"]
        for device_map in device_maps:
            model_args = ModelArguments(
                model_name_or_path="meta-llama/llama-3.2-1b",
                tokenizer_name_or_path="meta-llama/llama-3.2-1b",
                device_map=device_map,
                dtype="float32",
            )

            mock_model.return_value = MagicMock(spec=AutoModelForCausalLM)
            mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
            mock_tokenizer.return_value.pad_token = None
            mock_tokenizer.return_value.eos_token = "<|endoftext|>"

            model, _ = get_model_and_tokenizer(model_args)

            # Verify model loading with correct device map
            mock_model.assert_called_with(
                model_args.model_name_or_path,
                device_map=device_map,
                use_cache=False,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2"
                if device_map == "cuda"
                else None,
            )

    @patch("transformers.AutoModelForCausalLM.from_pretrained")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_get_model_and_tokenizer_chat_template(self, mock_tokenizer, mock_model):
        # Test chat template handling
        chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
        model_args = ModelArguments(
            model_name_or_path="meta-llama/llama-3.2-1b",
            tokenizer_name_or_path="meta-llama/llama-3.2-1b",
            device_map="cpu",
            dtype="float32",
            chat_template=chat_template,
        )

        mock_model.return_value = MagicMock(spec=AutoModelForCausalLM)
        mock_tokenizer.return_value = MagicMock(spec=AutoTokenizer)
        mock_tokenizer.return_value.pad_token = None
        mock_tokenizer.return_value.eos_token = "<|endoftext|>"

        _, tokenizer = get_model_and_tokenizer(model_args)

        # Verify chat template is set
        assert tokenizer.chat_template == chat_template

    @patch("revise.sft.load_dataset")
    def test_get_dataset(self, mock_load_dataset):
        # Mock dataset
        mock_dataset = Dataset.from_dict(
            {"question": ["What is 2+2?", "What is 3+3?"], "answer": ["4", "6"]}
        )
        mock_load_dataset.return_value = DatasetDict(
            {"train": mock_dataset, "test": mock_dataset}
        )

        # Mock tokenizer
        tokenizer = MagicMock(spec=AutoTokenizer)

        def fake_apply_chat_template(*args, **kwargs):
            return {
                "input_ids": torch.tensor([1, 2, 3]),
                "assistant_masks": torch.tensor([0, 0, 1]),
            }

        tokenizer.apply_chat_template = fake_apply_chat_template

        # Test dataset preparation
        train_dataset, eval_dataset = get_dataset(tokenizer)

        # Verify dataset loading
        mock_load_dataset.assert_called_once_with("openai/gsm8k", name="main")
        assert len(train_dataset) == 2
        assert len(eval_dataset) == 2

    @patch("revise.sft.load_dataset")
    def test_get_dataset_empty(self, mock_load_dataset):
        # Test with empty dataset
        mock_dataset = Dataset.from_dict({"question": [], "answer": []})
        mock_load_dataset.return_value = DatasetDict(
            {"train": mock_dataset, "test": mock_dataset}
        )

        tokenizer = MagicMock(spec=AutoTokenizer)
        tokenizer.apply_chat_template = MagicMock(
            return_value={
                "input_ids": torch.tensor([]),
                "assistant_masks": torch.tensor([]),
            }
        )

        train_dataset, eval_dataset = get_dataset(tokenizer)
        assert len(train_dataset) == 0
        assert len(eval_dataset) == 0

    @patch("revise.sft.get_model_and_tokenizer")
    @patch("revise.sft.get_dataset")
    def test_main(self, mock_get_dataset, mock_get_model_and_tokenizer, sft_args):
        # Mock model and tokenizer
        mock_model = MagicMock(spec=AutoModelForCausalLM)
        mock_model.to = MagicMock()
        mock_model.forward = MagicMock()
        mock_tokenizer = MagicMock(spec=AutoTokenizer)
        mock_tokenizer.encode = MagicMock(return_value=[0, 1, 2])
        mock_get_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)

        class DummyParam:
            def __init__(self, numel, requires_grad=True):
                self._numel = numel
                self.requires_grad = requires_grad

            def numel(self):
                return self._numel

        mock_model.parameters = MagicMock(
            return_value=[DummyParam(10), DummyParam(20, False)]
        )

        # Mock dataset
        mock_dataset = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "assistant_masks": [[0, 0, 1], [0, 0, 1]],
            }
        )
        mock_get_dataset.return_value = (mock_dataset, mock_dataset)

        # Mock trainer
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.train.return_value = MagicMock(metrics={"loss": 0.5})
        mock_trainer.save_model = MagicMock()
        mock_trainer.log_metrics = MagicMock()
        mock_trainer.save_metrics = MagicMock()
        mock_trainer.save_state = MagicMock()

        # Test main function
        with patch("revise.sft.Accelerator") as mock_accelerator, patch(
            "revise.sft.Trainer", return_value=mock_trainer
        ), patch("revise.sft.logger"):
            mock_accelerator.return_value.is_main_process = True
            mock_accelerator.return_value.main_process_first.return_value.__enter__.return_value = None
            main(sft_args)

            # Verify trainer calls
            mock_trainer.train.assert_called_once()
            mock_trainer.save_model.assert_called_once()
            mock_trainer.log_metrics.assert_called_once()
            mock_trainer.save_metrics.assert_called_once()
            mock_trainer.save_state.assert_called_once()

    @patch("revise.sft.get_model_and_tokenizer")
    @patch("revise.sft.get_dataset")
    def test_main_error_handling(
        self, mock_get_dataset, mock_get_model_and_tokenizer, sft_args
    ):
        # Test error handling in main function
        mock_get_model_and_tokenizer.side_effect = Exception("Model loading failed")

        with pytest.raises(Exception) as exc_info:
            with patch("revise.sft.Accelerator") as mock_accelerator:
                mock_accelerator.return_value.is_main_process = True
                main(sft_args)

        assert str(exc_info.value) == "Model loading failed"

    @patch("revise.sft.get_model_and_tokenizer")
    @patch("revise.sft.get_dataset")
    def test_main_training_error(
        self, mock_get_dataset, mock_get_model_and_tokenizer, sft_args
    ):
        # Test training error handling
        mock_model = MagicMock(spec=AutoModelForCausalLM)
        mock_tokenizer = MagicMock(spec=AutoTokenizer)
        mock_tokenizer.encode = MagicMock(return_value=[0, 1, 2])
        mock_get_model_and_tokenizer.return_value = (mock_model, mock_tokenizer)

        class DummyParam:
            def __init__(self, numel, requires_grad=True):
                self._numel = numel
                self.requires_grad = requires_grad

            def numel(self):
                return self._numel

        mock_model.parameters = MagicMock(
            return_value=[DummyParam(10), DummyParam(20, False)]
        )

        mock_dataset = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3], [4, 5, 6]],
                "assistant_masks": [[0, 0, 1], [0, 0, 1]],
            }
        )
        mock_get_dataset.return_value = (mock_dataset, mock_dataset)

        # Mock trainer that raises an error during training
        mock_trainer = MagicMock(spec=Trainer)
        mock_trainer.train.side_effect = Exception("Training failed")

        with pytest.raises(Exception) as exc_info:
            with patch("revise.sft.Accelerator") as mock_accelerator, patch(
                "revise.sft.Trainer", return_value=mock_trainer
            ), patch("revise.sft.logger"):
                mock_accelerator.return_value.is_main_process = True
                main(sft_args)

        assert str(exc_info.value) == "Training failed"
