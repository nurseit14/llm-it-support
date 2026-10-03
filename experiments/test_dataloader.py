import sys
from pathlib import Path

from torch.utils.data import DataLoader


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_PATH = PROJECT_ROOT / "src"

sys.path.append(
    str(SRC_PATH)
)


# ============================================================
# IMPORT PROJECT CODE
# ============================================================

from model import load_tokenizer

from dataset import (
    ITSupportDataset,
    create_collate_fn
)


# ============================================================
# DATASET PATH
# ============================================================

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.jsonl"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = load_tokenizer()


# ============================================================
# CREATE DATASET
# ============================================================

dataset = ITSupportDataset(

    TRAIN_PATH,

    tokenizer,

    max_length=512
)


# ============================================================
# CREATE DATALOADER
# ============================================================

dataloader = DataLoader(

    dataset,

    batch_size=4,

    shuffle=True,

    collate_fn=create_collate_fn(
        tokenizer
    )
)


# ============================================================
# INFORMATION
# ============================================================

print(
    "Dataset size:",
    len(dataset)
)


print(
    "Number of batches:",
    len(dataloader)
)


# ============================================================
# INSPECT FIRST BATCH
# ============================================================

for batch_number, batch in enumerate(
    dataloader,
    start=1
):


    print(
        f"\n========== "
        f"BATCH {batch_number} "
        f"=========="
    )


    print(
        "\nInput IDs shape:"
    )

    print(
        batch["input_ids"].shape
    )


    print(
        "\nAttention mask shape:"
    )

    print(
        batch["attention_mask"].shape
    )


    print(
        "\nLabels shape:"
    )

    print(
        batch["labels"].shape
    )


    print(
        "\nInput IDs:"
    )

    print(
        batch["input_ids"]
    )


    print(
        "\nAttention mask:"
    )

    print(
        batch["attention_mask"]
    )


    print(
        "\nLabels:"
    )

    print(
        batch["labels"]
    )


    # Only inspect first batch
    break