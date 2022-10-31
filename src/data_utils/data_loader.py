import random

from types import SimpleNamespace
from typing import List
from tqdm import tqdm
from copy import copy, deepcopy
from functools import lru_cache

from .load_race import load_race
from .load_auto_questions import load_autoq_train, load_autoq_test, load_autoq_test_filt
from .load_alta_b2 import load_B2_first, load_B2_first_schools
from .load_alta_mc4 import load_alta_MC4_data
from .load_alta_mc3 import load_alta_MC3_data

from ..utils.torch_utils import load_tokenizer, load_seq2seq_tokenizer

class DataLoader:
    #== Data loading utils ========================================================================#
    @classmethod
    def load_split(cls, data_name:str, mode:str, lim=None):
        split_index = {'train':0, 'dev':1, 'test':2}
        data = cls.load_data(data_name, lim)[split_index[mode]]
        return data
    
    @classmethod
    @lru_cache(maxsize=5)
    def load_data(cls, data_name:str, lim=None):
        data_fns = {'race':load_race,
                    'b2_first':load_B2_first,
                    'b2_first_school':load_B2_first_schools,
                    'alta_MC4':load_alta_MC4_data,
                    'alta_MC3':load_alta_MC3_data,
                    'auto_train':load_autoq_train,
                    'auto_test':load_autoq_test,
                    'auto_test_filt':load_autoq_test_filt}
        
        train, dev, test = data_fns[data_name]()
        
        if lim:
            train = cls.rand_select(train, lim)
            dev   = cls.rand_select(dev, lim)
            test  = cls.rand_select(test, lim)
        return train, dev, test

    @staticmethod
    def rand_select(data:list, lim:None):
        if data is None: return None
        
        random.seed(1)
        data = data.copy()
        random.shuffle(data)
        return data[:lim]
    
class QaDataLoader(DataLoader):
    def __init__(self, trans_name:str, formatting:str='standard'):
        self.tokenizer = load_tokenizer(trans_name)
        self.formatting = formatting
        
    #== MCRC Data processing (i.e. tokenizing text) ================================================#
    def prep_MCRC_split(self, data_name:str, mode:str, lim=None):
        data = self.load_split(data_name, mode, lim)
        return self._prep_MCRC_ids(data)
    
    def prep_MCRC_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_MCRC_ids(split) for split in [train, dev, test]]
        return train, dev, test
    
    def _prep_MCRC_ids(self, split_data):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            Q_ids   = self.tokenizer(ex.question).input_ids
            C_ids   = self.tokenizer(ex.context).input_ids
            options = [self.tokenizer(option).input_ids for option in ex.options]
            ex.input_ids = self._prep_inputs(Q_ids, C_ids, options)
        return split_data
            
    def _prep_inputs(self, Q_ids:List[int], C_ids:List[int], options:List[List[int]]):
        if self.formatting == 'standard':
            ids = [C_ids + Q_ids[1:-1] + O_ids[1:] for O_ids in options]
        elif self.formatting == 'O':
            ids = [O_ids for O_ids in options]
        elif self.formatting == 'QO':
            ids = [Q_ids[:-1] + O_ids[1:] for O_ids in options]
        elif self.formatting == 'CO':
            ids = [C_ids + O_ids[1:] for O_ids in options]
        return ids
    
class QgDataLoader(DataLoader):
    def __init__(self, trans_name:str, formatting:str='standard'):
        self.tokenizer = load_seq2seq_tokenizer(trans_name)
        self.formatting = formatting
        
    #== Question Generation Data processing (i.e. tokenizing text) ================================#
    def prep_QG_split(self, data_name:str, mode:str, lim=None):
        data = self.load_split(data_name, mode, lim)
        return self._prep_MCRC_ids(data)
    
    def prep_QG_data(self, data_name, lim=None):
        train, dev, test = self.load_data(data_name=data_name, lim=lim)
        train, dev, test = [self._prep_QG_ids(split) for split in [train, dev, test]]
        return train, dev, test
    
    def _prep_QG_ids(self, split_data):
        split_data = deepcopy(split_data)
        for ex in tqdm(split_data):
            answer = ex.options[ex.answer]
            Q_ids  = self.tokenizer(ex.question).input_ids
            C_ids  = self.tokenizer(ex.context).input_ids
            A_ids  = self.tokenizer(answer).input_ids
            
            # get input and reference ids
            input_ids = self._prep_inputs(Q_ids, C_ids)
            ex.input_ids = input_ids
            ex.label_ids = A_ids
        return split_data

    def _prep_inputs(self, Q_ids:List[int], C_ids:List[int]):
        if self.formatting == 'standard':
            ids = C_ids + Q_ids[1:]
        elif self.formatting == 'Q':
            ids = Q_ids
        elif self.formatting == 'C':
            ids = C_ids        
        return ids
    
    def prep_ex(self, ex:SimpleNamespace):
        ex = copy(ex)
        answer = ex.options[ex.answer]
        Q_ids  = self.tokenizer(ex.question).input_ids
        C_ids  = self.tokenizer(ex.context).input_ids
        A_ids  = self.tokenizer(answer).input_ids
        input_ids = self._prep_inputs(Q_ids, C_ids)
        ex.input_ids = input_ids
        ex.label_ids = A_ids
        return ex