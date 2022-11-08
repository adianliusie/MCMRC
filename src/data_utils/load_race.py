import os 
import time
import pickle

from datasets import load_dataset
from types import SimpleNamespace
from tqdm import tqdm

from ..utils.general import save_pickle, load_pickle, load_json, get_base_dir

def load_race():
    #load RACE-M and RACE-H data from hugginface
    race_m = load_dataset("race", "middle")
    race_h = load_dataset("race", "high")
    
    #load and format each train, validation and test split
    SPLITS = ['train', 'validation', 'test']
    train_m, dev_m, test_m = [format_race(race_m[split], 'M') for split in SPLITS]
    train_h, dev_h, test_h = [format_race(race_h[split], 'H') for split in SPLITS]

    #combine for output
    train = train_m + train_h 
    dev   = dev_m   + dev_h   
    test  = test_m  + test_h
    return train, dev, test    

def format_race(data, char):
    """ converts dict to SimpleNamespace for QA data"""
    outputs = []
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
    for k, ex in enumerate(data):
        ex_id = f'{char}_{k}'
        question = ex['question']
        context  = ex['article']
        options  = ex['options']
        answer   = ans_to_id[ex['answer']]
        ex_obj = SimpleNamespace(ex_id=ex_id, 
                                 question=question, 
                                 context=context, 
                                 options=options, 
                                 answer=answer)
        outputs.append(ex_obj)
    return outputs