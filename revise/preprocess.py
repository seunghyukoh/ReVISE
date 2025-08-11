import os
import re

from datasets import load_dataset
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()

    HUB_USER_ID = os.getenv("HUB_USER_ID")

    # Example preprocessing pipeline
    dataset = load_dataset("openai/gsm8k", name="main")
    dataset.cleanup_cache_files()

    dataset = dataset.map(
        lambda x: {
            "question": x["question"],
            "answer": re.sub(r"<<.*?>>", "", x["answer"]),
        }
    )

    dataset = dataset.map(
        lambda x: {"answer": x["answer"].replace("####", "The answer is:")}
    )

    train_eval_split = dataset["train"].train_test_split(test_size=0.1)
    dataset["train"] = train_eval_split["train"]
    dataset["eval"] = train_eval_split["test"]

    dataset.push_to_hub(f"{HUB_USER_ID}/gsm8k")
