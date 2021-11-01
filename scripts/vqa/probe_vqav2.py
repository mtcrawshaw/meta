"""
Script to probe VQAv2 dataset for questions involving spatial reasoning.

Note: This script should be run from the root of this repository.
"""

import time

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from vqav2 import VQAv2


BATCH_SIZE = 256
NUM_WORKERS = 8
VQAV2_PATH = "./scripts/vqa/data"
CUDA = True

# Keywords for spatial reasoning. The value for each keyword corresponds to whether or
# not we can reasonably augment the image/question input.
SPATIAL_KEYWORDS = {
    "aboard": False,
    "above": True,
    "across": False,
    "against": False,
    "ahead": False,
    "along": False,
    "alongside": False,
    "amid": False,
    "among": False,
    "amongst": False,
    "apart": False,
    "around": False,
    "aside": False,
    "away": False,
    "behind": False,
    "below": True,
    "beneath": True,
    "beside": True,
    "between": False,
    "beyond": False,
    "close": False,
    "down": False,
    "far": False,
    "inside": False,
    "into": False,
    "left": True,
    "near": False,
    "next": False,
    "onto": False,
    "over": True,
    "right": True,
    "toward": False,
    "under": True,
    "underneath": True,
    "up": False,
    "within": False,
}


def main() -> None:
    """ Main function to run VQAv2 probe. """

    # Start timer.
    start = time.time()

    # Load VQAv2 dataset.
    print("Loading dataset.")
    dataset = VQAv2(root=VQAV2_PATH, split="train")
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    # Set device.
    device = torch.device("cuda:0" if CUDA else "cpu")

    # Convert keywords into indices according to VQAv2 vocab.
    keyword_tensor = torch.zeros(len(SPATIAL_KEYWORDS)).long()
    for i, keyword in enumerate(SPATIAL_KEYWORDS.keys()):
        keyword_idx = dataset.token_to_index[keyword]
        keyword_tensor[i] = keyword_idx

    # Find and count questions with spatial keywords.
    print("Probing dataset.")
    keyword_questions = {keyword: [] for keyword in SPATIAL_KEYWORDS.keys()}
    counts = {keyword: 0 for keyword in SPATIAL_KEYWORDS.keys()}
    for batch in tqdm(loader):

        # Sample batch.
        _, questions, _, _, _, q_lengths = batch
        batch_size = questions.shape[0]
        questions = questions.to(device)
        q_lengths = q_lengths.to(device)

        # Check whether batch questions contain keywords, save questions that do, and
        # add them to counts.
        for i, keyword_idx in enumerate(keyword_tensor):
            keyword = dataset.index_to_token[int(keyword_idx)]
            contains = torch.any(questions == keyword_idx, dim=-1)

            contain_questions = questions[contains]
            contain_q_lengths = q_lengths[contains]
            natural_questions = dataset.decode_questions(
                contain_questions, contain_q_lengths
            )
            keyword_questions[keyword].extend(natural_questions)

            counts[keyword] += int(torch.sum(contains))

    # Print summary.
    num_spatial_questions = sum(counts[keyword] for keyword in SPATIAL_KEYWORDS.keys())
    num_augmentable_questions = sum(
        counts[keyword]
        for keyword, augmentable in SPATIAL_KEYWORDS.items()
        if augmentable
    )
    print("\nDone!")
    print(f"Total questions: {len(dataset)}")
    print(f"Spatial questions: {num_spatial_questions}")
    print(f"Augmentable questions: {num_augmentable_questions}")
    for keyword in SPATIAL_KEYWORDS.keys():
        print(f"Questions containing {keyword}: {counts[keyword]}")
    print("\nSample questions:")
    for keyword in SPATIAL_KEYWORDS.keys():
        print("")
        for question in keyword_questions[keyword]:
            print(question)

    # Print out execution time.
    end = time.time()
    run_time = end - start
    print(f"\nExecution time: {run_time:.3f}s")


if __name__ == "__main__":
    main()
