from types import SimpleNamespace
from ..utils.general import load_json, get_base_dir

BASE_DIR = get_base_dir()
generated_dir = f'{BASE_DIR}/data/generated_questions/t5_generated_racePlusPlus/beam4'

def load_autoq_test():
    path = f'{generated_dir}/on_test/fake_vs_real/fake.json'
    data = process_json_file(path)
    return None, None, data

def load_autoq_test_filt():
    path = f'{generated_dir}/on_test/filtered/filter_rate3/filtered.json'
    data = process_json_file(path)
    return None, None, data

def load_autoq_train():
    path = f'{generated_dir}/on_train/fake_vs_real/fake.json'
    data = process_json_file(path)
    return None, None, data

def process_json_file(path):
    data = load_json(path)
    
    outputs = []
    for k, ex in enumerate(data):
        question = ex['question']
        context  = ex['context']
        options  = ex['options']
        answer   = ex['label']
        ex_obj = SimpleNamespace(ex_id=k, 
                                 question=question, 
                                 context=context, 
                                 options=options, 
                                 answer=answer)
        if len(options) == 4:
            outputs.append(ex_obj)
    return outputs
