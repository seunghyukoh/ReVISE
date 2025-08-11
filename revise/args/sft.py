from dataclasses import dataclass, field
from typing import Optional

from trl import SFTConfig as DefaultSFTConfig


@dataclass
class SFTConfig(DefaultSFTConfig):
    run_name: str = field(
        default="sft",
        metadata={"help": "The name of the run"},
    )
    tags: list[str] = field(
        default_factory=list,
        metadata={"help": "The tags of the run"},
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "The model name or path to use for training"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer name or path to use for training"},
    )
    dataset_path: str = field(
        default=None, metadata={"help": "The dataset path to use for training"}
    )
    dataset_name: Optional[str] = field(
        default="default", metadata={"help": "The dataset name to use for training"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.dataset_path is None:
            raise ValueError("dataset_path is required")
        if self.dataset_name is None:
            raise ValueError("dataset_name is required")
