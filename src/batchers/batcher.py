import torch
import random

from itertools import islice
from typing import List
from types import SimpleNamespace

class Batcher():
    def __init__(self, max_len:int):
        self.device     = torch.device('cpu')
        self.max_len    = max_len

    def batches(self, data:list, bsz:int, shuffle:bool=False):
        """splits the data into batches and returns them"""
        data = data.copy()
        if shuffle: random.shuffle(data)
        batches = [examples[i:i+bsz] for i in range(0,len(examples), bsz)]
        for batch in batches:
            yield self.batchify(batch)
  
    def batchify(self, batch:List[list]):
        """each input is input ids and mask for utt, + label"""
        sample_id, input_ids, reference_ids = zip(*batch)  
        input_ids, attention_mask = self._get_padded_ids(input_ids)
        reference_ids, _ = self._get_padded_ids(reference_ids, pad_id=-100)
        return SimpleNamespace(sample_id=sample_id, 
                               input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               reference_ids=reference_ids)

    def to(self, device:torch.device):
        """ sets the device of the batcher """
        self.device = device
    
    def _get_padded_ids(self, ids:list, pad_id=0)->("pad_ids", "pad_mask"):
        """ pads ids to be flat """
        max_len = max([len(x) for x in ids])
        padded_ids = [x + [pad_id]*(max_len-len(x)) for x in ids]
        mask = [[1]*len(x) + [0]*(max_len-len(x)) for x in ids]
        ids = torch.LongTensor(padded_ids).to(self.device)
        mask = torch.FloatTensor(mask).to(self.device)
        return ids, mask

    def __call__(self, data, bsz, shuffle=False):
        """routes the main method do the batches function"""
        return self.batches(data=data, bsz=bsz, shuffle=shuffle)
    