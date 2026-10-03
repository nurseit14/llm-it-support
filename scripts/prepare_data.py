import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "it_support.jsonl"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

TRAIN_PATH = PROCESSED_DIR / "train.jsonl"
VAL_PATH = PROCESSED_DIR / "validation.jsonl"


# Create directory if it does not exist
PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# -----------------------------------------
# Load examples
# -----------------------------------------

examples = []

with open(RAW_DATA_PATH, "r", encoding="utf-8") as file:
    for line in file:

        if line.strip():
            example = json.loads(line)
            examples.append(example)


print(f"Total examples: {len(examples)}")


# -----------------------------------------
# Shuffle
# -----------------------------------------

random.seed(42)
random.shuffle(examples)


# -----------------------------------------
# Train / validation split
# -----------------------------------------

split_index = int(len(examples) * 0.8)

train_examples = examples[:split_index]
validation_examples = examples[split_index:]


print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(validation_examples)}")


# -----------------------------------------
# Save helper
# -----------------------------------------

def save_jsonl(examples, path):

    with open(path, "w", encoding="utf-8") as file:

        for example in examples:

            json.dump(
                example,
                file,
                ensure_ascii=False
            )

            file.write("\n")


# -----------------------------------------
# Save datasets
# -----------------------------------------

save_jsonl(
    train_examples,
    TRAIN_PATH
)

save_jsonl(
    validation_examples,
    VAL_PATH
)


print("\nSaved:")
print(TRAIN_PATH)
print(VAL_PATH)