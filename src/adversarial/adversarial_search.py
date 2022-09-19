import torch
import numpy as np
import random

from types import SimpleNamespace
from typing import List
from tqdm import tqdm
from functools import lru_cache

from ..trainers.QA_system_loader import SystemLoader
from ..utils.torch_utils import no_grad
from ..data_utils.data_loader import QaDataLoader
from ..batchers.QA_batcher import QaBatcher

class AdversarialOptionSearcher(SystemLoader):
    def __init__(self, exp_path, device=None):
        super().__init__(exp_path)
        self.set_up_helpers(device)
        self.convert_dataloader()
            
    def convert_dataloader(self):
        self.data_loader.__class__ = AdversarialOptionDataLoader
        
    @no_grad
    def find_adversarial_options(self, data_name, lim=None ,num_adv=None):
        prep_inputs, options, num_q = self.data_loader(data_name, lim, num_adv)
        
        matrix = np.zeros((num_q, len(options)+1))
        
        for ex in prep_inputs:
            input_ids = ex.input_ids.to(self.device)
            h = self.model.electra(input_ids)[0]
            pooled_output = self.model.sequence_summary(h)
            logit_score = self.model.classifier(pooled_output)
            matrix[ex.q_num, ex.opt_num] = logit_score
        return matrix, options
        
class AdversarialOptionDataLoader(QaDataLoader):
    """ Finds options which universally get selected by the system"""
    
    def __call__(self, data_name, lim=None, num_adv=None):
        train, dev, test = self.load_data(data_name)
        dev = self.rand_select(dev, lim) 
        
        #select random 5000 (num_adv) answers
        option_list = self.get_all_answers(train)
        if num_adv:
            rng = random.Random(1)
            option_list = rng.sample(option_list, num_adv)
        option_list = list(set(option_list))
        
        #produce the actual model inputs
        tok_model_inputs = self.prep_adv_search(dev, option_list)
        return tok_model_inputs, option_list, len(dev)
        
    def prep_adv_search(self, data, options):
        """given questions and options, creates all permutations for model inputs"""
        tokenized_inputs = []
        for q_num, ex in tqdm(enumerate(data), total=len(data)):
            Q_ids = self.tokenizer(ex.question).input_ids
            C_ids = self.cache_tokenize(ex.context)
            answer = ex.options[ex.answer]
            
            for k, option_text in enumerate([answer]+options):
                O_ids = self.tokenizer(option_text).input_ids
                ids = self._prep_single_option(Q_ids, C_ids, O_ids)
                if len(ids) > 512: 
                    ids = [ids[0]] + ids[-511:]
                input_ids = torch.LongTensor([ids])
                ex = SimpleNamespace(q_num=q_num, opt_num=k, input_ids=input_ids)
                yield ex
                
    def _prep_single_option(self, Q_ids:List[int], C_ids:List[int], O_ids:List[int]):
        if self.formatting == 'standard':
            ids = C_ids + Q_ids[1:-1] + O_ids[1:]
        elif self.formatting == 'O':
            ids = O_ids
        elif self.formatting == 'QO':
            ids = Q_ids[:-1] + O_ids[1:] 
        elif self.formatting == 'CO':
            ids = C_ids + O_ids[1:]
        return ids
    
    @staticmethod
    def get_all_options(data):
        all_options = set()
        for ex in data:
            all_options.update(ex.options)
        return list(all_options)

    @staticmethod
    def get_all_answers(data):
        all_options = []
        for ex in data:
            answer = ex.options[ex.answer]
            all_options.append(answer)
        return all_options
    
    @lru_cache(maxsize=1000000)
    def cache_tokenize(self, text):
        return self.tokenizer(text).input_ids
       