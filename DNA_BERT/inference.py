import torch
from multitask_dnabert import MultiTaskDNABERT
from data_utils import DoInference

CROSS_VALID = "CN1_2995.csv"
MODEL_NAME_TOKENIZER = "zhihan1996/DNA_bert_6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load model, you can download it locally via
#https://drive.google.com/uc?export=download&id=17xNnHWqdKvy2pAI_mkVF_qfSaQvBNcKv

model = torch.load("DNA_BERT_NANOPORE.pt", map_location=device)
model.eval()
model.to(device)

# 1. instantiate inference 
infer = DoInference(model, CROSS_VALID, MODEL_NAME_TOKENIZER)

#batches each sequence into 512bp subsequences and runs inference on each one.  It returns the classification for chromosome and telomere based on the average logits over all subsequences.  This architecture cannot support larger than 512 char tokens.
infer.test()