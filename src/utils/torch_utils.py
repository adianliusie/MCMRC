# Adian Liusie, 2022-SEP-02

import torch
from typing import Callable
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from transformers import ElectraTokenizerFast, ElectraForMultipleChoice

#== General Util Methods ===================================================================================
def no_grad(func:Callable)->Callable:
    """ decorator which detaches gradients """
    def inner(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return inner

#== Methods for loading tokenizer ==========================================================================
def load_tokenizer(system:str)->'Tokenizer':
    if system=='electra_base'   : tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")
    elif system=='electra_large': tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-large-discriminator")
    else: raise ValueError(f"{system} is an invalid system: no tokenizer found")
    return tokenizer

def load_seq2seq_tokenizer(system:str)->'Tokenizer':
    """ downloads and returns the relevant pretrained seq2seq tokenizer from huggingface """
    if   system == 't5_small' : tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    elif system == 't5_base'  : tokenizer = T5TokenizerFast.from_pretrained("t5-base")
    elif system == 't5_large' : tokenizer = T5TokenizerFast.from_pretrained("t5-large")
    else: raise ValueError(f"{system} is an invalid system: no seq2seq tokenizer found")
    return tokenizer

#== Methods for loading the base transformers ==============================================================
def load_MC_transformer(system:str)->'Model':
    """ downloads and returns miltiple choice transformers"""
    elec_base_path, elec_large_path =  "google/electra-base-discriminator", "google/electra-large-discriminator"
    if   system=='electra_base' : trans_model = ElectraForMultipleChoice.from_pretrained(elec_base_path, return_dict=True)
    elif system=='electra_large': trans_model = ElectraForMultipleChoice.from_pretrained(elec_large_path, return_dict=True)
    else: raise ValueError(f"{system} is an invalid system: no tokenizer found")
    return trans_model

def load_seq2seq_transformer(system:str)->'Model':
    """ downloads and returns the relevant pretrained seq2seq transformer from huggingface """
    if   system == 't5_small' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-small", return_dict=True)
    elif system == 't5_base'  : trans_model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    elif system == 't5_large' : trans_model = T5ForConditionalGeneration.from_pretrained("t5-large", return_dict=True)
    else: raise ValueError(f"{system} is an invalid system: no seq2seq model found")
    return trans_model
