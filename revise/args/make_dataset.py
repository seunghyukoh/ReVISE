from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MakeDatasetArgs:
    is_for_verifier_training: bool = field(
        default=False,
        metadata={"help": "Whether to make dataset for verifier training"},
    )
    model_name_or_path: str = field(
        default="meta-llama/llama-3.2-1b-instruct",
        metadata={"help": "The model name or path to use for training"},
    )
    dataset_type: str = field(
        default="gsm8k",
        metadata={"help": "The type of dataset to use for training"},
    )
    dataset_path: str = field(
        default=None,
        metadata={"help": "The dataset path to use for training"},
    )
    dataset_name: Optional[str] = field(
        default="default", metadata={"help": "The dataset name to use for training"}
    )
    output_dir: str = field(
        default="./tmp/make_dataset_output",
        metadata={"help": "The output directory to save the dataset"},
    )
    seed: int = field(default=42, metadata={"help": "The seed to use for training"})
    train_num_completions: int = field(
        default=10, metadata={"help": "The number of completions to use for training"}
    )
    eval_num_completions: int = field(
        default=1, metadata={"help": "The number of completions to use for evaluation"}
    )
    train_num_examples: Optional[int] = field(
        default=-1, metadata={"help": "The number of examples to use for training"}
    )
    eval_num_examples: Optional[int] = field(
        default=-1, metadata={"help": "The number of examples to use for evaluation"}
    )
    question_key: str = field(
        default="question", metadata={"help": "The key to use for the question"}
    )
    answer_key: str = field(
        default="answer", metadata={"help": "The key to use for the answer"}
    )
    prediction_key: str = field(
        default="prediction", metadata={"help": "The key to use for the prediction"}
    )
    hub_repo_id: str = field(
        default=None,
        metadata={"help": "The hub repository id to save the dataset"},
    )
    hub_config_name: str = field(
        default="default",
        metadata={"help": "The hub repository config name to save the dataset"},
    )
    num_splits: int = field(
        default=1,
        metadata={"help": "The number of splits to make the dataset"},
    )
    split_index: int = field(
        default=0,
        metadata={"help": "The index of the split to make the dataset"},
    )
