import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

if __name__ == "__main__":
    dataset = load_dataset("openai/gsm8k", name="main")
    dataset = dataset.map(
        lambda x: {
            "prompt": f"{x['question']}\nThink step by step and then answer.",
            "completion": x["answer"],
        }
    )

    # Note: Effective batch size := num_gpus * per_device_batch_size * grad_acc_steps
    training_args = SFTConfig(
        output_dir="./tmp",
        completion_only_loss=True,
        per_device_train_batch_size=4,  # per_device_batch_size
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # grad_acc_steps
        ddp_find_unused_parameters=False,
        bf16=True,
        max_steps=10,
        logging_steps=1,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=1,
        eval_strategy="steps",
        eval_steps=5,
        load_best_model_at_end=True,
        hub_strategy="end",
        push_to_hub=True,
        hub_model_id="JakeOh/llama-3.2-1b-sft-gsm8k",
        hub_private_repo=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/llama-3.2-1b",
        attn_implementation="flash_attention_2",  # Enable Flash Attention
        torch_dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-3.2-1b")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
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
