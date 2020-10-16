#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertForSequenceClassification, 
                          BertTokenizer, 
                          AdamW, 
                          BertConfig, 
                          get_linear_schedule_with_warmup,
                         )
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class FinansRequest(BaseModel):
    complaint: str

class FinansResponse(BaseModel):
    category: str

def classify(complaint):
    
    # Load a trained model and vocabulary that you have fine-tuned
    output_dir = '../text_classification/model_save/'
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    target_names = ['hesap','iade', 'iptal','kredi', 'kredi-karti', 'musteri-hizmetleri']

    # Tokenize all of the sequences and map the tokens to thier IDs.
    input_ids_new = []
    attention_masks_new = []

    encoded_dict = tokenizer.encode_plus(
                        complaint,                             # Sequence to encode
                        add_special_tokens = True,       # Add '[CLS]' and '[SEP]'
                        max_length = 128,                
                        padding = 'max_length',          # Pad and truncate
                        truncation=True,                 #Truncate the seq
                        return_attention_mask = True,    # Construct attn. masks
                        return_tensors = 'pt',           # Return pytorch tensors
                    )

    # Add the encoded sequences to the list    
    input_ids_new.append(encoded_dict['input_ids'])

    # And its attention mask
    attention_masks_new.append(encoded_dict['attention_mask'])

    input_ids_new = torch.cat(input_ids_new, dim=0)
    attention_masks_new = torch.cat(attention_masks_new, dim=0)


    # Prediction on test set
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids_new, token_type_ids=None, attention_mask=attention_masks_new)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy() 
        predictions = logits[0].tolist() 

    category_name = target_names[predictions.index(max(predictions))]
    print("The predicted category is:")
    print(target_names[predictions.index(max(predictions))])
    
    return(category_name)


@app.post("/predict", response_model=FinansResponse)
def predict(request: FinansRequest):
   category_name = classify(request.complaint)
   return FinansResponse(
       category=category_name
   )
   



