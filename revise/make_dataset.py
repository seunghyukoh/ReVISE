import inspect
import json
import os
from typing import Callable, Union

from datasets import Dataset, load_dataset

from revise.config import DEFAULT_EOS_TOKEN, DEFAULT_REFINE_TOKEN
from revise.evaluators.comparison_evaluator import (
    BaseComparisonEvaluator,
    GSM8KEvaluator,
)
from revise.generators.vllm_generator import VllmGenerationParams, VllmGenerator
from revise.prompts import prepare_batch_chat_messages_fns, prepare_chat_messages_fns
from revise.utils import configure_logging, hash_params

logger = configure_logging(level="info")


def generate_and_evaluate(
    model_path: str,
    dataset: Dataset,
    evaluator: Union[BaseComparisonEvaluator],
    prepare_batch_chat_messages_fn: Callable,
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
    refine_token=DEFAULT_REFINE_TOKEN,
    eos_token=DEFAULT_EOS_TOKEN,
):
    params = dict(
        model_path=model_path,
        dataset=str(dataset),
        evaluator_source=inspect.getsource(evaluator.__class__),
        prepare_batch_chat_messages_source=inspect.getsource(
            prepare_batch_chat_messages_fn
        ),
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

    # Load dataset
    if question_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{question_key}' column.")
    if answer_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{answer_key}' column.")

    if num_examples > 0:
        dataset = dataset.select(range(num_examples))

    # Make chat messages
    batch_chat_messages = prepare_batch_chat_messages_fn(dataset=dataset)

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
    outputs = generator.generate(batch_chat_messages)  # List[List[str]]

    # Evaluate
    new_dataset = []

    questions = dataset[question_key]
    gt_answers = dataset[answer_key]

    def routine(pred_trajectory: str):
        pred_trajectory = pred_trajectory.split(eos_token)[0]
        predictions = pred_trajectory.split(refine_token)
        predictions = [
            prediction.strip() for prediction in predictions if prediction.strip()
        ]
        return predictions

    for question, gt_answer, pred_trajectories in zip(questions, gt_answers, outputs):
        num_trajectories = len(pred_trajectories)  # List[str]
        if num_trajectories < num_completions:
            logger.warning(
                f"Only {num_trajectories} predictions generated for {question}"
            )

        if num_trajectories == 0:
            logger.warning(f"No predictions generated for {question}")
            continue

        for pred_trajectory in pred_trajectories:
            predictions = routine(pred_trajectory)
            num_predictions = len(predictions)

            scores = evaluator.run(
                answers=[gt_answer] * num_predictions,
                predictions=predictions,
                return_results=True,
            )
            score_list = scores["score_list"]

            for prediction, is_correct in zip(predictions, score_list):
                new_dataset.append(
                    {
                        question_key: question,
                        answer_key: gt_answer,
                        prediction_key: prediction,
                        "is_correct": is_correct,
                    }
                )
                if not is_correct:
                    break

    new_dataset = Dataset.from_list(new_dataset)
    with open(cache_path, "w") as f:
        data = new_dataset.to_list()
        for line in data:
            f.write(json.dumps(line) + "\n")

    return new_dataset


def make_dataset(
    dataset,
    question_key,
    answer_key,
    prediction_key,
    prepare_chat_messages_fn: Callable,
    is_for_verifier_training=False,
    use_gt: bool = False,
    refine_token=DEFAULT_REFINE_TOKEN,
):
    if question_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{question_key}' column.")

    if answer_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{answer_key}' column.")

    if prediction_key not in dataset.column_names:
        raise ValueError(f"Dataset must contain a '{prediction_key}' column.")

    if "is_correct" not in dataset.column_names:
        raise ValueError("Dataset must contain a 'is_correct' column.")

    new_dataset = []

    for sample in dataset:
        question = sample[question_key]
        gt_answer = sample[answer_key]

        prediction = sample[prediction_key]
        prediction = prediction.split(refine_token)[-1]

        is_correct = sample["is_correct"]

        if is_correct:
            chosen_message = prediction
            rejected_message = prediction + refine_token
        else:
            chosen_message = prediction + refine_token
            if not is_for_verifier_training:
                chosen_message += gt_answer
            rejected_message = prediction

        user_messages = prepare_chat_messages_fn(question)
        chosen = user_messages + [{"role": "assistant", "content": chosen_message}]
        rejected = user_messages + [{"role": "assistant", "content": rejected_message}]

        new_dataset.append(
            {
                question_key: question,
                answer_key: gt_answer,
                prediction_key: prediction,
                "is_correct": is_correct,
                "is_verifier": is_for_verifier_training,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    if use_gt:
        # Use gt answer as prediction
        for sample in dataset:
            question = sample[question_key]
            gt_answer = sample[answer_key]

            chosen_message = gt_answer
            rejected_message = gt_answer + refine_token

            user_messages = prepare_chat_messages_fn(question)
            chosen = user_messages + [{"role": "assistant", "content": chosen_message}]
            rejected = user_messages + [
                {"role": "assistant", "content": rejected_message}
            ]

            new_dataset.append(
                {
                    question_key: question,
                    answer_key: gt_answer,
                    prediction_key: gt_answer,
                    "is_correct": True,
                    "is_verifier": is_for_verifier_training,
                    "is_gt": True,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    new_dataset = Dataset.from_list(new_dataset)
    return new_dataset


if __name__ == "__main__":
    from datasets import DatasetDict
    from revise.prompts import prepare_batch_chat_messages_fns
    from transformers.hf_argparser import HfArgumentParser

    from revise.args.make_dataset import MakeDatasetArgs

    parser = HfArgumentParser(MakeDatasetArgs)
    [args] = parser.parse_args_into_dataclasses()

    assert args.dataset_type in ["gsm8k"], "Only gsm8k is supported for now"

    evaluator = GSM8KEvaluator(mode="flexible")

    dataset = load_dataset(args.dataset_path, args.dataset_name)
    train_dataset = dataset["train"].select(
        range(args.split_index, len(dataset["train"]), args.num_splits)
    )
    eval_dataset = dataset["eval"]

    prepare_batch_chat_messages_fn = prepare_batch_chat_messages_fns[args.dataset_type]
    prepare_chat_messages_fn = prepare_chat_messages_fns[args.dataset_type]

    new_train_dataset_evaluated = generate_and_evaluate(
        model_path=args.model_name_or_path,
        evaluator=evaluator,
        dataset=train_dataset,
        prepare_batch_chat_messages_fn=prepare_batch_chat_messages_fn,
        num_examples=args.train_num_examples,
        num_completions=args.train_num_completions,
        seed=args.seed,
        question_key=args.question_key,
        answer_key=args.answer_key,
        prediction_key=args.prediction_key,
    )

    new_eval_dataset_evaluated = generate_and_evaluate(
        model_path=args.model_name_or_path,
        evaluator=evaluator,
        dataset=eval_dataset,
        prepare_batch_chat_messages_fn=prepare_batch_chat_messages_fn,
        num_examples=args.eval_num_examples,
        num_completions=args.eval_num_completions,
        seed=args.seed,
        question_key=args.question_key,
        answer_key=args.answer_key,
        prediction_key=args.prediction_key,
    )

    new_train_dataset = make_dataset(
        dataset=new_train_dataset_evaluated,
        question_key=args.question_key,
        answer_key=args.answer_key,
        prediction_key=args.prediction_key,
        prepare_chat_messages_fn=prepare_chat_messages_fn,
        is_for_verifier_training=args.is_for_verifier_training,
    )

    new_eval_dataset = make_dataset(
        dataset=new_eval_dataset_evaluated,
        question_key=args.question_key,
        answer_key=args.answer_key,
        prediction_key=args.prediction_key,
        prepare_chat_messages_fn=prepare_chat_messages_fn,
        is_for_verifier_training=args.is_for_verifier_training,
    )

    new_dataset = DatasetDict(
        {
            "train": new_train_dataset,
            "eval": new_eval_dataset,
        }
    )

    new_dataset.save_to_disk(args.output_dir)

    if args.hub_repo_id is not None:
        new_dataset.push_to_hub(args.hub_repo_id, config_name=args.hub_config_name)
