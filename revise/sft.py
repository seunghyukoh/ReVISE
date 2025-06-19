import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import SFTTrainer

from revise.args.sft import SFTConfig


def load_dataset(path: str, name: str):
    from datasets import load_dataset

    dataset = load_dataset(path, name=name)
    dataset = dataset.map(
        lambda x: {
            "prompt": f"{x['question']}\nThink step by step and then answer.",
            "completion": x["answer"],
        }
    )

    return {"train": dataset["train"], "eval": dataset["test"]}


if __name__ == "__main__":
    parser = HfArgumentParser(SFTConfig)
    [training_args] = parser.parse_args_into_dataclasses()

    dataset = load_dataset(training_args.dataset_path, training_args.dataset_name)

    model = AutoModelForCausalLM.from_pretrained(
        training_args.model_name_or_path,
        attn_implementation="flash_attention_2",  # Enable Flash Attention
        torch_dtype="bfloat16",
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path or training_args.model_name_or_path
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"] if "eval" in dataset else None,
        args=training_args,
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
