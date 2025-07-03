from datasets import load_dataset
import re

if __name__ == "__main__":
    HUB_USER_ID = "JakeOh"

    # Example preprocessing pipeline
    gsm8k = load_dataset("openai/gsm8k", name="main")
    gsm8k = gsm8k.map(
        lambda x: {
            "question": x["question"],
            "answer": re.sub(r"<<.*?>>", "", x["answer"]),
        }
    )

    train_eval_split = gsm8k["train"].train_test_split(test_size=0.1)
    gsm8k["train"] = train_eval_split["train"]
    gsm8k["eval"] = train_eval_split["test"]

    gsm8k.push_to_hub(f"{HUB_USER_ID}/gsm8k")
