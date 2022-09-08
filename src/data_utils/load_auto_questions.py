from types import SimpleNamespace
from ..utils.general import save_pickle, load_pickle, load_json, get_base_dir

BASE_DIR = get_base_dir()
generated_dir = f'{BASE_DIR}/data/generated_questions/t5_generated_racePlusPlus/beam4'

def load_automatic_questions_test():
    path = f'{generated_dir}/on_test/fake_vs_real/fake.json'
    data = process_json_file(path)
    return data, data, data

def load_automatic_questions_train():
    path = f'{generated_dir}/on_train/fake_vs_real/fake.json'
    data = process_json_file(path)
    return data, data, data
    
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
        outputs.append(ex_obj)
    return outputs
