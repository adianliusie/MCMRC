import csv
import pandas as pd
import numpy as np

from types import SimpleNamespace
from typing import List

from src.utils.general import get_base_dir
from .load_alta_mc4 import get_q_headers, process_cand_dist

def load_alta_MC3_data():
    BASE_DIR = get_base_dir()
    data_path = f'{BASE_DIR}/data/alta/MCQ_v3/3optionMCQ_Stats (Final).csv'
    
    #load data and convert from weird windows format
    df = pd.read_csv(data_path, encoding='cp1252')
    df = df.replace({'’': "'"}, regex=True)
    df = df.replace({'‘': "'"}, regex=True)
    df = df.replace({'–':'-'}, regex=True)
    data = format_csv(df)
    return None, None, data

def format_csv(df:pd.DataFrame)->List[SimpleNamespace]:
    ans_to_id = {'A':0, 'B':1, 'C':2}
    q_headers = get_q_headers(df)

    output = []
    for index, row in df.iterrows():
        context = row['Text']
        q_headers = [q for q in q_headers if not pd.isna(row[q])]
        for q_num in q_headers:
            #get exam type
            exam_type = row['Product'].lower()

            # Get specific question details
            ex_id = f'{index}-{q_num}'
            question = row[q_num]
            options  = [row[f'{q_num}{i}'] for i in ['A', 'B', 'C']]
            answer   = row[f'{q_num}_answer']
            answer   = ans_to_id[answer]

            # Get candidates chosen answer distribution if valid
            cand_dist   = [row[f'{q_num}_distract_{i}_fac'] for i in ['a', 'b', 'c']]
            disc_scores = [row[f'{q_num}_distract_{i}_disc'] for i in ['a', 'b', 'c']] 
            cand_dist   = process_cand_dist(cand_dist, disc_scores)

            #process output type
            ex_obj = SimpleNamespace(
                         ex_id=ex_id, 
                         question=question, 
                         context=context, 
                         options=options, 
                         answer=answer,
                         exam_type=exam_type,
                         cand_dist=cand_dist,
                     )
            output.append(ex_obj)     
    return output
