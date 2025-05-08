import hashlib
import json
import logging
import os

from datasets import Dataset, load_dataset
from evaluators.comparison_evaluator import GSM8KEvaluator
from generators.vllm_generator import VllmGenerationParams, VllmGenerator
from transformers import AutoTokenizer

import inspect


def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    return logger


def hash_params(params):
    return hashlib.sha256(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_dataset(
    model_path,
    dataset,
    evaluator,
    prepare_prompts_fn,
    max_new_tokens=1024,
    temperature=0.7,
    num_completions=10,
    top_p=0.9,
    num_examples=-1,
    seed=42,
    question_key="question",
    answer_key="answer",
    prediction_key="prediction",
    ignore_cache=False,
):
    params = dict(
        model_path=model_path,
        dataset=str(dataset),
        evaluator_source=inspect.getsource(evaluator.__class__),
        prepare_prompts_source=inspect.getsource(prepare_prompts_fn),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_completions=num_completions,
        top_p=top_p,
        num_examples=num_examples,
        seed=seed,
        question_key=question_key,
        answer_key=answer_key,
        prediction_key=prediction_key,
    )
    hashed = hash_params(params)
    cache_path = f"./.cache/{hashed}.jsonl"
    os.makedirs(".cache", exist_ok=True)
    if os.path.exists(cache_path) and not ignore_cache:
        logger.info(f"Loading dataset from cache: {cache_path}")
        with open(cache_path, "r") as f:
            return Dataset.from_list([json.loads(line) for line in f])

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load dataset
    train_dataset = dataset["train"]
    if question_key not in train_dataset.column_names:
        raise ValueError(f"Dataset must contain a '{question_key}' column.")
    if answer_key not in train_dataset.column_names:
        raise ValueError(f"Dataset must contain a '{answer_key}' column.")

    if num_examples > 0:
        train_dataset = train_dataset.select(range(num_examples))

    # Make prompts
    prompts = prepare_prompts_fn(train_dataset, tokenizer)

    # Get generator
    generation_params = VllmGenerationParams(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        num_completions=num_completions,
        top_p=top_p,
        seed=seed,
        skip_special_tokens=False,
    )
    generator = VllmGenerator(model=model_path, gen_params=generation_params)

    # Generate
    predictions = generator.generate(prompts)  # List[List[str]]

    # Evaluate
    new_dataset = []

    questions = train_dataset[question_key]
    gt_answers = train_dataset[answer_key]
    for question, gt_answer, prediction_set in zip(questions, gt_answers, predictions):
        num_predictions = len(prediction_set)
        scores = evaluator.run(
            answers=[gt_answer] * num_predictions,
            predictions=prediction_set,
            return_results=True,
        )
        score_list = scores["score_list"]

        for prediction, is_correct in zip(prediction_set, score_list):
            new_dataset.append(
                {
                    question_key: question,
                    answer_key: gt_answer,
                    prediction_key: prediction,
                    "is_correct": is_correct,
                }
            )

    new_dataset = Dataset.from_list(new_dataset)
    with open(cache_path, "w") as f:
        data = new_dataset.to_list()
        for line in data:
            f.write(json.dumps(line) + "\n")

    return new_dataset


if __name__ == "__main__":
    from prompts import prepare_prompts_fns

    logger = configure_logging()

    # Get evaluator
    make_dataset(
        model_path="meta-llama/llama-3.2-1b-instruct",
        evaluator=GSM8KEvaluator(mode="flexible"),
        dataset=load_dataset("openai/gsm8k", name="main"),
        prepare_prompts_fn=prepare_prompts_fns["gsm8k"],
        num_examples=100,
        num_completions=2,
    )
