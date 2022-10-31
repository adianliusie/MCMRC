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
    #race_c = load_race_c()
    
    #load and format each train, validation and test split
    SPLITS = ['train', 'validation', 'test']
    train_m, dev_m, test_m = [format_race(race_m[split], 'M') for split in SPLITS]
    train_h, dev_h, test_h = [format_race(race_h[split], 'H') for split in SPLITS]
    #train_c, dev_c, test_c = [format_race(race_c[split], 'C') for split in SPLITS]

    #combine for output
    train = train_m + train_h #+ train_c
    dev   = dev_m   + dev_h   #+ dev_c
    test  = test_m  + test_h  #+ test_c
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

def load_race_c():
    BASE_DIR = get_base_dir()
    race_c_path = f'{BASE_DIR}/data/RACE-C'
    pickle_path = f'{BASE_DIR}/data/RACE-C/cache.pkl'
    splits_path = [f'{race_c_path}/{split}' for split in ['train', 'dev', 'test']]
    
    # Download data if missing
    if not os.path.isdir(race_c_path):
        raise Exception('need to implement automatic data downloading')
    
    # Load cached data if exists, otherwise process and cache
    if os.path.isfile(pickle_path):
        train, dev, test = load_pickle(pickle_path)
    else:
        train, dev, test = [load_race_c_split(path) for path in splits_path]
        save_pickle(data=[train, dev, test], path=pickle_path)
        
    return {'train':train, 'validation':dev, 'test':test}

def load_race_c_split(split_path:str):
    file_paths = [f'{split_path}/{f_path}' for f_path in os.listdir(split_path)]
    outputs = []
    for file_path in file_paths:
        outputs += load_race_file(file_path)
    return outputs

def load_race_file(path:str):
    file_data = load_json(path)
    article = file_data['article']
    answers = file_data['answers']
    options = file_data['options']
    questions = file_data['questions']
    
    outputs = []
    assert len(questions) == len(options) == len(answers)
    for k in range(len(questions)):
        ex = {'question':questions[k], 
              'article':article,
              'options':options[k],
              'answer':answers[k]}
        outputs.append(ex)
    return outputs
