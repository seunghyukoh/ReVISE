from dataclasses import dataclass, field

from trl import DPOConfig as DefaultDPOConfig


@dataclass
class DPOConfig(DefaultDPOConfig):
    model_name_or_path: str = field(
        default=None, metadata={"help": "The model name or path to use for training"}
    )
    tokenizer_name_or_path: str = field(
        default=None,
        metadata={"help": "The tokenizer name or path to use for training"},
    )
    dataset_path: str = field(
        default=None, metadata={"help": "The dataset path to use for training"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "The dataset name to use for training"}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.model_name_or_path is None:
            raise ValueError("model_name_or_path is required")
        if self.tokenizer_name_or_path is None:
            raise ValueError("tokenizer_name_or_path is required")
        if self.dataset_path is None:
            raise ValueError("dataset_path is required")
        if self.dataset_name is None:
            raise ValueError("dataset_name is required")
