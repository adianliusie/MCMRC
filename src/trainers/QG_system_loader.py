import torch
import numpy as np
import os

from tqdm import tqdm
from typing import List
from types import SimpleNamespace

from .QG_trainer import QgTrainer
from ..utils.torch_utils import no_grad
from ..utils.dir_helper import DirHelper
from ..data_utils.data_loader import QaDataLoader

class QgSystemLoader(QgTrainer):
    """Base loader class- the inherited class inherits
       the Trainer so has all experiment methods"""

    def __init__(self, exp_path:str, device=None):
        self.dir = DirHelper.load_dir(exp_path)
        self.set_up_helpers(device)
        
    def set_up_helpers(self, device=None):
        #load training arguments and set up helpers
        args = self.dir.load_args('model_args.json')
        super().set_up_helpers(args)

        #load final model
        self.load_model()
        self.model.eval()
        
        self.device = device if device else 'cuda:0'
        self.to(self.device)

    def generate_option(self, ex=None):
        assert type(ex)==SimpleNamespace or type(ex)==str
        
        tokenizer = self.data_loader.tokenizer
        if type(ex)==str: input_ids = tokenizer(input_text).input_ids 
        else            : input_ids = self.data_loader.prep_ex(ex).input_ids
            
        input_ids = torch.LongTensor([input_ids]).to(self.device)
        model_output = self.model.generate(input_ids=input_ids)
        answer = tokenizer.decode(model_output[0], skip_special_tokens=True)
        return answer
    
    