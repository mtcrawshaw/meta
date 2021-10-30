"""
Demo script to run LXMERT on image/question pairs.

Note: This script should be run from the root of this repository.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from transformers import LxmertTokenizer, LxmertForQuestionAnswering

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config, get_data
from vqav2 import VQAv2

BATCH_SIZE = 4
NUM_WORKERS = 4

EXAMPLE_IMG_PATH = "./scripts/vqa/vqa_example.jpg"
VQAV2_PATH = "./scripts/vqa/data"
VQA_ANSWERS_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
EXAMPLE_QUESTIONS = [
    "what food is on the plate?",
    "what color is the wall?",
    "what is on the tray?",
    "what is next to the tea pot?",
    "are there any people here?",
]


def main(example=False) -> None:
    """ Main function to run LXMERT demo. """

    # Load LXMERT model and dictionary to interpret LXMERT outputs.
    lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
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
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "visual_feats": features,
            "visual_pos": normalized_boxes,
            "token_type_ids": inputs.token_type_ids,
        }

    else:

        # Load VQAv2 dataset.
        dataset = VQAv2(root=VQAV2_PATH, split="train")
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        )

        # Sample question and visual features.
        features, questions, answers, bboxes, _, q_lengths = next(iter(loader))
        batch_size = features.shape[0]
        features = features.squeeze(2)
        features = features.transpose(1, 2)
        bboxes = bboxes.transpose(1, 2)

        # Collect inputs to LXMERT.
        max_length = questions.shape[1]
        attention_mask = torch.stack([torch.arange(max_length)] * batch_size)
        attention_mask = (attention_mask < q_lengths.unsqueeze(-1)).long()
        lxmert_kwargs = {
            "input_ids": questions,
            "attention_mask": attention_mask,
            "visual_feats": features,
            "visual_pos": bboxes,
            "token_type_ids": torch.zeros_like(questions),
        }

        # Convert questions to natural language for display.
        natural_questions = dataset.decode_questions(questions, q_lengths)

    # Run inference on questions.
    output = lxmert(output_attentions=False, **lxmert_kwargs,)

    # Show questions and answers.
    for i in range(batch_size):
        prediction = vqa_answers[output["question_answering_score"][i].argmax(-1)]
        if answers is not None:
            ground_truth = vqa_answers[answers[i].argmax(-1)]
        else:
            ground_truth = "Unknown"
        print(f"Question: {natural_questions[i]}")
        print(f"Predicted Answer: {prediction}")
        print(f"Actual Answer: {ground_truth}")
        print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example",
        action="store_true",
        default=False,
        help="Use example image and hard-coded example questions.",
    )
    args = parser.parse_args()

    main(**vars(args))
