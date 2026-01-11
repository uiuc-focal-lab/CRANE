from syncode import Syncode, AdaptiveSynCode
from syncode.language_model import KeywordsStoppingCriteria
import time
from .gsm_symbolic_constraints import generate_gsm_symbolic_with_itergen
from .fol_constraints import generate_fol_with_itergen
from crane.main import IterGen, CRANE


class BaseLM():
    def __init__(self, model_name, mode = 'grammar_strict', grammar = 'json', 
                 max_tokens=512, temperature=0.0, device = 'cuda', stop_words = None, task = 'spider',
                 recurrence_penalty:float=1.0, max_itergen_iter = 5, backwards_limit = 10, num_return_sequences = 1, 
                 start_symbol = "<<", start_in_grammar = True, end_symbol = ">>", end_in_grammar = True, **kwargs) -> None:
        

        if task == 'gsm_symbolic':
            if start_symbol != '<<' and end_symbol != '>>':
                with open("./itergen/iter_syncode/iter_syncode/parsers/grammars/gsm_custom_grammar.lark", "r") as file:
                    grammar = file.read() 
                    grammar = grammar.replace('[[START]]', ' '.join([f'"{x}"' for x in start_symbol])).replace('[[END]]', ' '.join([f'"{x}"' for x in end_symbol]))
        
        gen_kwargs = {**kwargs, **{'max_new_tokens': int(max_tokens)}}
        self.grammar = grammar
        temperature = float(temperature)
        if temperature == 0.0: 
            gen_kwargs['do_sample'] = False
        else:
            gen_kwargs['temperature'] = temperature
            gen_kwargs['do_sample'] = True
        
        if mode == 'itergen' or mode == 'adaptive':
            if mode == 'itergen':
                self.model = IterGen(grammar= grammar, 
                                    model_id=model_name, 
                                    parse_output_only=True, 
                                    recurrence_penalty=recurrence_penalty,
                                    stop_strings=stop_words, 
                                    device = device,
                                    **gen_kwargs)
            else:
                self.model = CRANE(grammar= grammar, 
                                    model_id=model_name, 
                                    parse_output_only=True, 
                                    recurrence_penalty=recurrence_penalty,
                                    stop_strings=stop_words, 
                                    device = device,
                                    start_symbol=start_symbol,
                                    start_in_grammar=start_in_grammar,
                                    end_symbol=end_symbol,
                                    end_in_grammar=end_in_grammar,
                                    **gen_kwargs)
            self._generate_with_itergen = eval(f'generate_{task}_with_itergen')
            self.max_itergen_iter = max_itergen_iter
            self.backwards_limit = backwards_limit
        else:
            if mode == 'adaptive_grammar':
                self.model = AdaptiveSynCode(
                    mode= 'grammar_strict',
                    model= model_name,
                    grammar= grammar,
                    parser = 'lr',
                    device = device, 
                    parse_output_only = True, 
                    start_symbol=start_symbol,
                    start_in_grammar=start_in_grammar,
                    end_symbol=end_symbol,
                    end_in_grammar=end_in_grammar,
                    **gen_kwargs
                )
            else:
                self.model = Syncode(
                    mode= mode,
                    model= model_name,
                    grammar= grammar,
                    parser = 'lr',
                    device = device, 
                    parse_output_only = True, 
                    **gen_kwargs
                )
        self.model_name = model_name
        self.stop_words = stop_words
        self.mode = mode
        self.num_return_sequences = num_return_sequences
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        self.task = task

    def add_start_end_symbol(self, batch, prompt_key): 
        if self.task == 'gsm_symbolic':
            for i in range(len(batch)): 
               if isinstance(batch[i][prompt_key], list):
                   batch[i][prompt_key] = [{**x, 'content': x['content'].replace('[[START]]', self.start_symbol).replace('[[END]]', self.end_symbol)} for x in batch[i][prompt_key]]
               else:
                   batch[i][prompt_key] = batch[i][prompt_key].replace('[[START]]', self.start_symbol).replace('[[END]]', self.end_symbol)
        return batch
    
    
    def __call__(self, batch, prompt_key = 'prompt', response_key = 'llm_response', info_key = 'response_info'):
        self.add_start_end_symbol(batch, prompt_key)
        if isinstance(self.model, Syncode):
            return self.generate_with_syncode(batch, prompt_key, response_key, info_key)
        
        elif isinstance(self.model, IterGen):
            return self.generate_with_itergen(batch, prompt_key, response_key, info_key)
        
        else:
            raise ValueError(f'Generation type {self.model} not recognized')

    
    def generate_with_itergen(self, batch, prompt_key = 'prompt', response_key = 'llm_response', info_key = 'response_info'):
        new_batch = []
        for problem in batch:
            for _ in range(self.num_return_sequences):
                try:
                    out, res_dct = self._generate_with_itergen(self.model, problem, prompt_key, self.max_itergen_iter, self.backwards_limit)
                except: 
                    out, res_dct = [""], {'time': 0, 'tokens': 0}
                new_batch.append(problem.copy())
                new_batch[-1][response_key] = out[0]
                new_batch[-1][info_key] = res_dct
        return new_batch        
    
    
    def generate_with_syncode(self, batch, prompt_key = 'prompt', response_key = 'llm_response', info_key = 'response_info'):
        prompts = [x[prompt_key] for x in batch]
        
        if isinstance(prompts[0], list):
            prompts = [self.model.model.tokenizer.apply_chat_template(prompt, tokenize = False, add_generation_prompt = True) for prompt in prompts]

        if batch[0].get('reasoning', None) is not None:
            prompts = [prompt + batch[i]['reasoning'] for i, prompt in enumerate(prompts)]
        
        new_batch = []
        for i, prompt in enumerate(prompts):
            for _ in range(self.num_return_sequences):
                start = time.time()
                #out = self.model.model.generate_batch_completion_grammar(prompt, 1, stop_words = self.stop_words)[0]
                out = self.model.model.generate_grammar_constrained_completion(prompt, 1, stop_words = self.stop_words)[0]
                dt = time.time() - start
                new_batch.append(batch[i].copy())
                new_batch[-1][info_key] = {'time': dt, 'tokens': len(self.model.model.tokenizer(out)['input_ids'])}
                new_batch[-1][response_key] = out
        return new_batch 



