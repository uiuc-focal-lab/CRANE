from .base import BasePrompter, BaseParser
from .fol import FOLParser
from .gsm_symbolic import GSMSymbolicParser




PARSE_MAP = {
    'fol': {'text': FOLParser, 'prover9': FOLParser}, 
    'gsm_symbolic': {'text': GSMSymbolicParser, 'gsm': GSMSymbolicParser},
}

def get_stop_words(dataset, do_cot, grammar_mode = None):
    if dataset == 'fol':
        if grammar_mode is not None and grammar_mode == 'prover9':
            return ['Note', '------']
        return [ '------']
    elif dataset == 'gsm_symbolic':
        if not do_cot:
            return ['\n</think>', '?']
        return ['\n</think>', '?']
    return None