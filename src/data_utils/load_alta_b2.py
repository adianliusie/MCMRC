import csv
import pandas as pd

from types import SimpleNamespace
from typing import List

from ..utils.general import get_base_dir

def load_B2_first():
    BASE_DIR = get_base_dir()
    data_path = f'{BASE_DIR}/data/alta/MCQ_debug/B2 Dataset (for upload)_B2First.csv'
    
    #load data
    df = pd.read_csv(data_path)
    data = format_csv(df)
    return None, None, data

def load_B2_first_schools():
    BASE_DIR = get_base_dir()
    data_path = f'{BASE_DIR}/data/alta/MCQ_debug/B2 Dataset (for upload)_B2First_for_Schools.csv'
    
    #load data
    df = pd.read_csv(data_path)
    data = format_csv(df)
    return None, None, data

def format_csv(df:pd.DataFrame)->List[SimpleNamespace]:
    """ converts pandas dataframe to list of """
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
    df2 = df.set_index('Task ID').T
    num_q = find_num_questions(df2)
    
    outputs = []
    for index, row in df2.iterrows():
        context = row['Text']
        for q_num in range(1, num_q+1):
            ex_id = f'{index}-{q_num}'
            question = row[f'Q{q_num}']
            options  = [row[f'Q{q_num}{i}'] for i in ['A', 'B', 'C', 'D']]

            answer   = row[f'Q{q_num} Key']
            answer   = ans_to_id[answer]
            ex_obj = SimpleNamespace(ex_id=ex_id, 
                                     question=question, 
                                     context=context, 
                                     options=options, 
                                     answer=answer)
            outputs.append(ex_obj)            
    return outputs

def find_num_questions(df2:pd.DataFrame)->int:
    question_keys = [x for x in df2.columns if x[-3:]=='Key']
    return len(question_keys)