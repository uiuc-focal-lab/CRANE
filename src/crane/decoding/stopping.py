"""Stopping criteria shared by cursor and baseline decoding."""
import torch
from transformers import StoppingCriteria

class KeywordsStoppingCriteria(StoppingCriteria):
    '''
    Assume batch_size = 1

    We can use this class to check if the stop word is present in the completion. This is more expensive since we need to decode the completion to check if the stop word is present.
    '''
    def __init__(self, tokenizer, stop_words = [], input_ids_cutoff = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.stop_words_ids = []
        self.input_ids_cutoff = input_ids_cutoff

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        partial_output = self.tokenizer.batch_decode(input_ids[:, self.input_ids_cutoff: ], skip_special_tokens=True)[0]
        for stop_word in self.stop_words:
            if stop_word == '?' and stop_word in partial_output:
                if 'The final answer is' in partial_output:
                    return True
            else:
                if stop_word in partial_output:
                    return True
        return False
