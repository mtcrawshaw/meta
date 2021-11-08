"""
Demo script to run LXMERT on image/question pairs.

Note: This script should be run from the root of this repository.
"""

import argparse

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import LxmertTokenizer, LxmertForQuestionAnswering

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config, get_data
from vqav2 import VQAv2


BATCH_SIZE = 128
NUM_WORKERS = 4
SEED = 0
LIMIT_SIZE = None

EXAMPLE_IMG_PATH = "./scripts/vqa/vqa_example.jpg"
VQAV2_PATH = "./scripts/vqa/data"
VQA_ANSWERS_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
EXAMPLE_QUESTIONS = [
    "what is this photo taken looking through?",
    "what position is this man playing?",
    "what color is the players shirt?",
    "is this man a professional baseball player?",
]


def main(example: bool = False, cuda: bool = False) -> None:
    """ Main function to run LXMERT demo. """

    # Set random seed.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Set device.
    device = torch.device("cuda:0" if cuda else "cpu")

    # Load LXMERT model and dictionary to interpret LXMERT outputs.
    lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
    lxmert = lxmert.to(device)
    vqa_answers = get_data(VQA_ANSWERS_URL)

    # Sample question and corresponding visual features.
    if example:

        # Load feature extractor and tokenizer.
        frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        frcnn = GeneralizedRCNN.from_pretrained(
            "unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg
        )
        image_preprocess = Preprocess(frcnn_cfg)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")

        # Run Faster-RCNN.
        images, sizes, scales_yx = image_preprocess(EXAMPLE_IMG_PATH)
        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=frcnn_cfg.max_detections,
            return_tensors="pt",
        )
        normalized_boxes = output_dict.get("normalized_boxes")
        features = output_dict.get("roi_features")

        # Tokenize questions.
        natural_questions = EXAMPLE_QUESTIONS
        answers = None
        inputs = tokenizer(
            natural_questions,
            padding="max_length",
            max_length=20,
            truncation=True,
            return_tensors="pt",
        )

        # Collect inputs to LXMERT.
        batch_size = len(EXAMPLE_QUESTIONS)
        lxmert_kwargs = {
            "input_ids": inputs.input_ids.to(device),
            "attention_mask": inputs.attention_mask.to(device),
            "visual_feats": features.to(device),
            "visual_pos": normalized_boxes.to(device),
            "token_type_ids": inputs.token_type_ids.to(device),
        }

        # Run inference on questions.
        output = lxmert(output_attentions=False, **lxmert_kwargs)

        # Show questions and answers.
        for i in range(batch_size):
            prediction = vqa_answers[output["question_answering_score"][i].argmax(-1)]
            print(f"Question: {natural_questions[i]}")
            print(f"Predicted Answer: {prediction}")
            print("")

    else:

        # Load VQAv2 dataset.
        dataset = VQAv2(root=VQAV2_PATH, split="train", limit_size=LIMIT_SIZE)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        )

        accuracies = []

        # Evaluate LXMERT on VQAv2.
        for batch in tqdm(loader):

            # Sample batch, move to device, and resize appropriately.
            features, questions, answers, bboxes, _, q_lengths = batch
            features = features.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            bboxes = bboxes.to(device)
            q_lengths = q_lengths.to(device)
            batch_size = features.shape[0]
            features = features.squeeze(2)
            features = features.transpose(1, 2)
            bboxes = bboxes.transpose(1, 2)

            # Collect inputs to LXMERT.
            max_length = questions.shape[1]
            attention_mask = torch.stack([torch.arange(max_length, device=q_lengths.device)] * batch_size)
            attention_mask = (attention_mask < q_lengths.unsqueeze(-1)).long()
            lxmert_kwargs = {
                "input_ids": questions,
                "attention_mask": attention_mask,
                "visual_feats": features,
                "visual_pos": bboxes,
                "token_type_ids": torch.zeros_like(questions),
            }

            # Get model prediction and compute accuracy.
            output = lxmert(output_attentions=False, **lxmert_kwargs)
            prediction = output["question_answering_score"]
            accuracy = torch.sum(prediction.argmax(dim=-1) == answers.argmax(dim=-1))
            accuracies.append(float(accuracy) / batch_size)

        print(f"Accuracy: {np.mean(accuracies)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        action="store_true",
        default=False,
        help="Use example image and hard-coded example questions.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        default=False,
        help="Use GPU.",
    )
    args = parser.parse_args()

    main(**vars(args))
