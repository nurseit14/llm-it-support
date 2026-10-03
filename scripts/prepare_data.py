import json
import random
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

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


TRAIN_PATH = (
    PROCESSED_DIR
    / "train.jsonl"
)


VAL_PATH = (
    PROCESSED_DIR
    / "validation.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.8

RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(

    parents=True,

    exist_ok=True
)


# ============================================================
# LOAD RAW DATA
# ============================================================

examples = []


with open(
    RAW_DATA_PATH,
    "r",
    encoding="utf-8"
) as file:


    for line in file:


        if line.strip():


            example = json.loads(
                line
            )


            examples.append(
                example
            )


print(
    f"Total examples: "
    f"{len(examples)}"
)


# ============================================================
# SHUFFLE
# ============================================================

random.seed(
    RANDOM_SEED
)


random.shuffle(
    examples
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(

    len(examples)

    * TRAIN_RATIO
)


train_examples = (
    examples[:split_index]
)


validation_examples = (
    examples[split_index:]
)


print(
    f"Training examples: "
    f"{len(train_examples)}"
)


print(
    f"Validation examples: "
    f"{len(validation_examples)}"
)


# ============================================================
# SAVE HELPER
# ============================================================

def save_jsonl(
    examples,
    path
):


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:


        for example in examples:


            json.dump(

                example,

                file,

                ensure_ascii=False
            )


            file.write("\n")


# ============================================================
# SAVE DATASETS
# ============================================================

save_jsonl(
    train_examples,
    TRAIN_PATH
)


save_jsonl(
    validation_examples,
    VAL_PATH
)


print(
    "\nDatasets saved:"
)


print(
    TRAIN_PATH
)


print(
    VAL_PATH
)