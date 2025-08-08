import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
)
from trl import SFTTrainer

from revise.args.sft import SFTConfig
from revise.config import DEFAULT_CHAT_TEMPLATE
from revise.prompts import prepare_chat_messages_fns


def load_dataset(path: str, name: str):
    from datasets import load_dataset

    dataset = load_dataset(path, name=name)
    prepare_chat_messages_fn = prepare_chat_messages_fns["gsm8k"]
    dataset = dataset.map(
        lambda x: {
            "prompt": prepare_chat_messages_fn(x["question"]),
            "completion": [
                {
                    "role": "assistant",
                    "content": x["answer"],
                }
            ],
        }
    )

    return {"train": dataset["train"], "eval": dataset["eval"]}


if __name__ == "__main__":
    parser = HfArgumentParser(SFTConfig)
    [training_args] = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        import os

        import wandb
        from dotenv import load_dotenv

        load_dotenv()

        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project="revise",
            name=training_args.run_name,
            config=training_args.to_dict(),
            tags=training_args.tags,
            group="sft",
        )

    dataset = load_dataset(training_args.dataset_path, training_args.dataset_name)

    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        attn_implementation="flash_attention_2",  # Enable Flash Attention
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path or training_args.model_name_or_path
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if "eval" in dataset else None,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_args.early_stopping_patience
            )
        ],
    )

    # Clear cache to mitigate memory issues
    torch.cuda.empty_cache()

    # Train the model
    train_result = trainer.train()

    # Save the model
    trainer.save_model()

    # Save the metrics of the training
    if hasattr(train_result, "metrics"):
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    # Save the state of the trainer
    trainer.save_state()
