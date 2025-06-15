import logging
import os
from pprint import pprint
from typing import Tuple

import torch
import wandb
from accelerate import Accelerator
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
)
from trl import DataCollatorForCompletionOnlyLM

from revise.args.parser import parse_args
from revise.args.sft import ModelArguments, SFTArguments

logger = logging.getLogger(__name__)
is_main_process = True


def get_model_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device_map = model_args.device_map
    if device_map is None:
        device_map = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    dtype = getattr(torch, model_args.dtype) if model_args.dtype is not None else None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=device_map,
        use_cache=False,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2" if device_map == "cuda" else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path or model_args.model_name_or_path,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.chat_template is not None:
        tokenizer.chat_template = model_args.chat_template

    return model, tokenizer


def get_dataset(tokenizer):
    # TODO: Update this to your own dataset.
    dataset = load_dataset("openai/gsm8k", name="main")

    def routine(sample):
        question = sample["question"]
        answer = sample["answer"]

        from revise.prompts import prepare_gsm8k_chat_messages

        messages = prepare_gsm8k_chat_messages(question)
        messages += [{"role": "assistant", "content": answer}]

        encoding = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "assistant_masks": encoding["assistant_masks"].squeeze(),
        }

    dataset = dataset.map(routine, batched=False)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    return train_dataset, eval_dataset


def main(args: SFTArguments):
    model_args = args.model_args
    training_args = args.training_args

    # Set seed
    set_seed(training_args.seed)

    # Load tokenizer and model
    with Accelerator().main_process_first():
        model, tokenizer = get_model_and_tokenizer(model_args)

    # Log model params
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    # Log model
    if is_main_process:
        pprint(model)

    # Load dataset
    with Accelerator().main_process_first():
        train_dataset, validation_dataset = get_dataset(tokenizer=tokenizer)

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM(
            response_template="<|assistant|>\n",
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    import torch._dynamo as d2

    d2.config.capture_scalar_outputs = True

    load_dotenv()

    logger.setLevel(logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Parse arguments
    args: SFTArguments = parse_args(SFTArguments)

    accelerator = Accelerator()

    is_main_process = accelerator.is_main_process
    if is_main_process:
        raw_args = args.to_dict()

        # Print arguments
        pprint(raw_args)

        wandb.init(
            id=args.uuid,
            project=os.getenv("WANDB_PROJECT", "revise"),
            name=args.training_args.run_name,
            config=raw_args,
            group="sft",
        )

    accelerator.wait_for_everyone()

    main(args)
