import torch
import numpy as np
import random
import torch.nn.functional as F

from types import SimpleNamespace
from typing import List
from tqdm import tqdm
from functools import lru_cache

from ..trainers.QA_system_loader import SystemLoader
from ..trainers.QG_system_loader import QgSystemLoader

from ..utils.torch_utils import no_grad
from ..data_utils.data_loader import QaDataLoader
from ..batchers.QA_batcher import QaBatcher
from ..utils.general import get_base_dir, join_paths

class ImposterSystemLoader(SystemLoader):
    def __init__(self, exp_path, device=None):
        super().__init__(exp_path)
        self.set_up_helpers(device)
        self.convert_dataloader()
            
    def convert_dataloader(self):
        self.data_loader.__class__ = ImposterOptionDataLoader
    
    def get_imposter_preds(self, imposter_path, data_name:str, mode='test', lim=None):
        probs = self.get_imposter_probs(imposter_path, data_name, mode, lim)
        preds = {}
        for ex_id, probs in probs.items():
            preds[ex_id] = int(np.argmax(probs, axis=-1))  
        return preds
        
    @no_grad
    def get_imposter_probs(self, imposter_path:str, data_name:str, mode='test', lim=None):
        """get imposter model predictions for given data"""
        self.model.eval()
        self.to(self.device)
        
        #this code mimics the internals of dataloader
        data = self.data_loader.load_imposter_data(imposter_path, data_name, mode, 
                                                   lim=lim, device=self.device)
        eval_data = self.data_loader._prep_MCRC_ids(data)
    
        eval_batches = self.batcher(data=eval_data, bsz=1, shuffle=False)        
        probabilties = {}
        for batch in tqdm(eval_batches):
            ex_id = batch.ex_id[0]
            output = self.model_output(batch)

            logits = output.logits.squeeze(0)
            if logits.shape and logits.shape[-1] > 1:  # Get probabilities of predictions
                prob = F.softmax(logits, dim=-1)
            probabilties[ex_id] = prob.cpu().numpy()
        return probabilties
    
class ImposterOptionDataLoader(QaDataLoader):
    #== Adversarial Imposter Option Evaluation ====================================================================
    def load_imposter_data(self, imposter_path, data_name, mode='test', lim=None, device='cuda'):
        imposter_system = QgSystemLoader(imposter_path, device=device)
        imposter_system.to(device)

        data = self.load_split(data_name, mode, lim)
        random.seed(1)        

        for ex in tqdm(data):
            #for each example replace a random option with the imposter option
            imposter_option = imposter_system.generate_option(ex=ex)
            rand_opt = random.choice([i for i in range(len(ex.options)) if i != ex.answer])
            ex.options[rand_opt] = imposter_option 
        return data
    