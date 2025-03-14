import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class MultiTaskDNABERT(nn.Module):
    def __init__(self, model_name, num_chr_labels, num_tel_labels, dropout=0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        # Two classification heads
        self.classifier_chr = nn.Linear(hidden_size, num_chr_labels)
        self.classifier_tel = nn.Linear(hidden_size, num_tel_labels)

    def forward(self, input_ids, attention_mask, labels_chr=None, labels_tel=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # if there's a pooler_output, use it, else fallback to [CLS]
        if outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        x = self.dropout(pooled)
        logits_chr = self.classifier_chr(x)
        logits_tel = self.classifier_tel(x)

        loss = None
        if labels_chr is not None and labels_tel is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_chr = loss_fct(logits_chr, labels_chr)
            loss_tel = loss_fct(logits_tel, labels_tel)
            loss = loss_chr + loss_tel

        return {
            'loss': loss,
            'logits_chr': logits_chr,
            'logits_tel': logits_tel
        }