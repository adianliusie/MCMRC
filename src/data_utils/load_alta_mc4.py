import csv
import pandas as pd
import numpy as np

from types import SimpleNamespace
from typing import List

from src.utils.general import get_base_dir

def load_alta_MC4_data():
    BASE_DIR = get_base_dir()
    data_path = f'{BASE_DIR}/data/alta/MCQ_v2/4optionMCQ_stats (Final).csv'
    
    #load data and convert from weird windows format
    df = pd.read_csv(data_path, encoding='cp1252')
    df = df.replace({'’': "'"}, regex=True)
    df = df.replace({'‘': "'"}, regex=True)
    df = df.replace({'–':'-'}, regex=True)
    data = format_csv(df)
    return None, None, data

def format_csv(df:pd.DataFrame)->List[SimpleNamespace]:
    ans_to_id = {'A':0, 'B':1, 'C':2, 'D':3}
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
            options  = [row[f'{q_num}{i}'] for i in ['A', 'B', 'C', 'D']]
            answer   = row[f'{q_num}_answer']
            answer   = ans_to_id[answer]

            # Get candidates chosen answer distribution if valid
            cand_dist   = [row[f'{q_num}_distract_{i}_fac'] for i in ['a', 'b', 'c', 'd']]
            disc_scores = [row[f'{q_num}_distract_{i}_disc'] for i in ['a', 'b', 'c', 'd']] 
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

def get_q_headers(df):
    """ gets all column headers for the question"""
    questions = [x.replace('A','') for x in df.columns if x[0]=='Q' and x[-1]=='A']
    return questions

def process_cand_dist(cand_dist, disc_scores):
    #if candidate distribution not provided, return None
    if pd.isna(cand_dist[0]):
        return None
    else:
        assert min(cand_dist) >= 0
        # if candidate distribution sum between 0.98 and 1.02, accept and normalise
        if abs(sum(cand_dist)-1.00)<0.021:
            output = np.array(cand_dist)/np.sum(cand_dist)
            return output
            
        else:
            #if one of the discriminative scores is 0, remove the score and see if it normalises
            min_disc, index = min([(abs(d), k) for k, d in enumerate(disc_scores)]) 
            if float(min_disc) == 0:
                temp_cand_dist = cand_dist.copy()
                temp_cand_dist[index] = 0
                if abs(sum(temp_cand_dist)-1.00)<0.021:
                    output = np.array(temp_cand_dist)/np.sum(temp_cand_dist)
                    return output
                
    # if all fails, then note distribution is invalid, but return the og dist
    return f'invalid dist: {cand_dist} sum={round(sum(cand_dist), 4)}'
