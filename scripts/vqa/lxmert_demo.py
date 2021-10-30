"""
Demo script to run LXMERT on image/question pairs.

Note: This script should be run from the root of this repository.
"""


from transformers import LxmertTokenizer, LxmertForQuestionAnswering

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config, get_data


IMG_PATH = "./scripts/vqa/vqa_example.jpg"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
QUESTIONS = [
    "what food is on the plate?",
    "what color is the wall?",
    "what is on the tray?",
    "what is next to the tea pot?",
    "are there any people here?",
]


# Load model.
frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")
vqa_answers = get_data(VQA_URL)

# Run Faster-RCNN.
images, sizes, scales_yx = image_preprocess(IMG_PATH)
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

# Run inference on questions.
inputs = tokenizer(
    QUESTIONS,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt",
)
print(f"input_ids: {inputs.input_ids}")
print(f"attention_mask: {inputs.attention_mask}")
print(f"visual_feats: {features.shape}")
print(f"visual_pos: {normalized_boxes.shape}")
print(f"token_type_ids: {inputs.token_type_ids}")
output = lxmert(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    visual_feats=features,
    visual_pos=normalized_boxes,
    token_type_ids=inputs.token_type_ids,
    output_attentions=False,
)
for i in range(len(QUESTIONS)):
    prediction = vqa_answers[output["question_answering_score"][i].argmax(-1)]
    print(f"Question: {QUESTIONS[i]}")
    print(f"Predicted Answers: {prediction}")
    print("")
