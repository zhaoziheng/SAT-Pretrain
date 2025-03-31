import torch
from typing import Union, List
from transformers import AutoTokenizer
import os
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MyTokenizer():
    def __init__(self, tokenizer='../others/pubmed_bert', max_length=256, random_truncate_prob=0):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.random_truncate_prob = random_truncate_prob
        
    def tokenize_to_fix_length(self, texts:[str, List[str]]) -> torch.LongTensor:
        """
        tokenize a lits of strings or a single string, pad/trunctate to max length input of the text tower

        Args:
            texts (str, List[str]]): a string 

        Returns:
            torch.LongTensor: the tokenized tensor and the attention mask(mask out paddings)
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = '[CLS]'
        eot_token = '[SEP]'
        all_token_ids = []
        for text in texts:  # a string
            tokens = [sot_token] + self.tokenizer.tokenize(text) + [eot_token]  # list of str
            all_token_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))  
        result = torch.zeros(len(all_token_ids), self.max_length, dtype=torch.long) # N, max_len

        for i, token_ids in enumerate(all_token_ids):   # list of int
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]  # Truncate
                token_ids[-1] = self.tokenizer.convert_tokens_to_ids('[SEP]')
            result[i, :len(token_ids)] = torch.tensor(token_ids)
            
        attn_mask = torch.where(result>0, 1, 0)
            
        return {'input_ids':result, 'attention_mask':attn_mask}
    
    def tokenize(self, texts:[str, List[str]], random_truncate:bool=False) -> torch.LongTensor:
        """
        tokenize a lits of strings or a single string, pad/trunctate to min(max length input of the text tower, max length of samples in this batch)

        Args:
            texts (str, List[str]]): a string 

        Returns:
            torch.LongTensor: the tokenized tensor and the attention mask(mask out paddings)
        """
        if isinstance(texts, str):
            texts = [texts]
            
        random_truncate_prob = self.random_truncate_prob if random_truncate else 0.0

        sot_token = '[CLS]'
        eot_token = '[SEP]'
        all_token_ids = []
        max_len_in_this_batch = 0
        for text in texts:  # a string
            tokens = [sot_token] + self.tokenizer.tokenize(text) + [eot_token]  # list of str
            if len(tokens) > max_len_in_this_batch:
                max_len_in_this_batch = len(tokens)
            all_token_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))  
        if max_len_in_this_batch > self.max_length:
            max_len_in_this_batch = self.max_length
        result = torch.zeros(len(all_token_ids), max_len_in_this_batch, dtype=torch.long) 

        for i, token_ids in enumerate(all_token_ids):   # list of int
            if len(token_ids) > max_len_in_this_batch:
                if random.random() < random_truncate_prob: # when def is too long, random truncate as an augmentation
                    start = random.randint(0, len(token_ids)-max_len_in_this_batch)
                    end = start + max_len_in_this_batch
                    token_ids = token_ids[start:end]
                else:
                    token_ids = token_ids[:max_len_in_this_batch]  # Truncate
                token_ids[-1] = self.tokenizer.convert_tokens_to_ids('[SEP]')
            result[i, :len(token_ids)] = torch.tensor(token_ids)
            
        attn_mask = torch.where(result>0, 1, 0)
        
        assert result.shape[1] < 512, result.shape
            
        return {'input_ids':result, 'attention_mask':attn_mask}