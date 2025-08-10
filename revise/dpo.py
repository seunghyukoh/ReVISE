import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from revise.args.dpo import DPOConfig
from revise.config import DEFAULT_CHAT_TEMPLATE, DEFAULT_REFINE_TOKEN
from revise.dpo_trainer import DPOTrainer


def load_dataset(path: str, name: str):
    from datasets import load_dataset

    dataset = load_dataset(path, name=name)

    return {"train": dataset["train"], "eval": dataset["eval"]}


if __name__ == "__main__":
    parser = HfArgumentParser(DPOConfig)
    [training_args] = parser.parse_args_into_dataclasses()

    if training_args.should_log:
        import os

        import wandb
        from dotenv import load_dotenv

        load_dotenv()

        wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project=os.getenv("WANDB_PROJECT_NAME"),
            name=training_args.run_name,
            config=training_args.to_dict(),
            tags=training_args.tags,
            group="dpo",
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

    refine_token_id = tokenizer.encode(DEFAULT_REFINE_TOKEN, add_special_tokens=False)[
        0
    ]

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        processing_class=tokenizer,
        refine_token_id=refine_token_id,
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
