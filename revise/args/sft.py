from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments as HfTrainingArguments

from revise.args.parser import BaseArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model name or path to use for training"}
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer name or path to use for training"},
    )

    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The template for the chat."},
    )

    device_map: Optional[str] = field(
        default=None,
        metadata={"help": "The device to use for training"},
    )

    dtype: Optional[str] = field(
        default=None,
        metadata={"help": "The dtype to use for training"},
    )

    # LoRA (Low Rank Adaptation)
    use_lora: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use Low Rank Adaptation",
        },
    )
    lora_rank: Optional[int] = field(
        default=16,
        metadata={
            "help": "Set the rank for Low Rank Adaptation",
        },
    )
    lora_alpha: Optional[float] = field(
        default=4,
        metadata={
            "help": "Set the alpha for Low Rank Adaptation",
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "Set the dropout for Low Rank Adaptation",
        },
    )


@dataclass
class TrainingArguments(HfTrainingArguments):
    pass


class SFTArguments(BaseArguments):
    ARG_COMPONENTS = [
        ModelArguments,
        TrainingArguments,
    ]

    model_args: ModelArguments
    training_args: TrainingArguments

    def __init__(
        self,
        model_args: ModelArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()

        assert model_args is not None
        assert training_args is not None

        self.model_args = model_args
        self.training_args = training_args

    def to_dict(self):
        return {
            "uuid": self.uuid,
            **self.model_args.__dict__,
            **self.training_args.__dict__,
        }
