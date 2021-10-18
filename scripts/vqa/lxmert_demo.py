""" Demo script to run LXMERT on image/question pairs. """


import torch
from transformers import LxmertTokenizer, LxmertForQuestionAnswering


tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")

question, text = "Who was Jim Henson", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors="pt")

outputs = model(**inputs)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
