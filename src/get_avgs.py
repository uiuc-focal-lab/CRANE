import srsly 
import numpy as np
import fire 
from lark import Lark
import syncode.evaluation.fol_eval as fol_eval

def get_avgs(path):
    res = srsly.read_jsonl(path)
    times = []
    tokens = []
    corrects = []
    for r in res:
        times.append(r['total_time'])
        tokens.append(r['total_tokens'])
        corrects.append(r['correct'])
    
    print('Mean Accuracy ', np.mean(corrects))
    print('Mean Tokens ', np.mean(tokens))
    print('Mean Time ', np.mean(times))
    return {'acc': round(100 * np.mean(corrects),2),
            'tokens': round(np.mean(tokens), 2),
            'time': round(np.mean(times), 2)
            }

def get_pass_k_fol(paths):
    res_lst = [srsly.read_jsonl(path) for path in paths]
    times = [[] for _ in range(3)]
    tokens = [[] for _ in range(3)]
    corrects = [[] for _ in range(3)]
    compiles = [[] for _ in range(3)]
    for samp_1, samp_2, samp_3 in zip(res_lst[0], res_lst[1], res_lst[2]):
        times[0].append(samp_1['total_time'])
        times[1].append(samp_1['total_time'] + samp_2['total_time'])
        times[2].append(samp_1['total_time'] + samp_2['total_time'] + samp_3['total_time'])
        
        tokens[0].append(samp_1['total_tokens'])
        tokens[1].append(samp_1['total_tokens'] + samp_2['total_tokens'])
        tokens[2].append(samp_1['total_tokens'] + samp_2['total_tokens'] + samp_3['total_tokens'])
        
        corrects[0].append(samp_1['correct'])
        corrects[1].append(samp_1['correct'] or samp_2['correct'])
        corrects[2].append(samp_1['correct'] or samp_2['correct'] or samp_3['correct'])
        
        compiles[0].append(samp_1['compiles'])
        compiles[1].append(samp_1['compiles'] or samp_2['compiles'])
        compiles[2].append(samp_1['compiles'] or samp_2['compiles'] or samp_3['compiles'])
    
    return {'acc': [round(100 * np.mean(corrects[i]),2) for i in range(3)],
            'compiles': [round(100 * np.mean(compiles[i]),2) for i in range(3)],
            'tokens': [round(np.mean(tokens[i]), 2) for i in range(3)],
            'time': [round(np.mean(times[i]), 2) for i in range(3)]}

def get_avgs_fol(path):
    res = srsly.read_jsonl(path)
    times = []
    tokens = []
    corrects = []
    compiles = []
    for r in res:
        times.append(r['total_time'])
        tokens.append(r['total_tokens'])
        corrects.append(r['correct'])
        compiles.append(r['compiles'])
    
    print('Mean Accuracy ', np.mean(corrects))
    print('Mean Compiles ', np.mean(compiles))
    print('Mean Tokens ', np.mean(tokens))
    print('Mean Time ', np.mean(times))
    return {'acc': round(100 * np.mean(corrects),2),
            'compiles': round(100 * np.mean(compiles),2),
            'tokens': round(np.mean(tokens), 2),
            'time': round(np.mean(times), 2)
            }   



def check_gsm_parsed(expr, parser):
    if expr == "" or '{' in expr or '}' in expr or 'round(' in expr:
        return False

    if not expr.startswith('<<') or not expr.endswith('>>'):
        return False
    
    try:
        parser.parse(expr)
        return True
    except:
        return False


def get_pass_k_gsm(paths, parser):
    res_lst = [srsly.read_jsonl(path) for path in paths]
    times = [[] for _ in range(3)]
    tokens = [[] for _ in range(3)]
    corrects = [[] for _ in range(3)]
    parses = [[] for _ in range(3)]
    for samp_1, samp_2, samp_3 in zip(res_lst[0], res_lst[1], res_lst[2]):
        times[0].append(samp_1['total_time'])
        times[1].append(samp_1['total_time'] + samp_2['total_time'])
        times[2].append(samp_1['total_time'] + samp_2['total_time'] + samp_3['total_time'])
        
        tokens[0].append(samp_1['total_tokens'])
        tokens[1].append(samp_1['total_tokens'] + samp_2['total_tokens'])
        tokens[2].append(samp_1['total_tokens'] + samp_2['total_tokens'] + samp_3['total_tokens'])
        
        corrects[0].append(samp_1['correct'])
        corrects[1].append(samp_1['correct'] or samp_2['correct'])
        corrects[2].append(samp_1['correct'] or samp_2['correct'] or samp_3['correct'])
        
        parses[0].append(check_gsm_parsed(samp_1['parsed_completion'], parser))
        parses[1].append(check_gsm_parsed(samp_1['parsed_completion'], parser) or check_gsm_parsed(samp_2['parsed_completion'], parser))
        parses[2].append(check_gsm_parsed(samp_1['parsed_completion'], parser) or check_gsm_parsed(samp_2['parsed_completion'], parser) or check_gsm_parsed(samp_3['parsed_completion'], parser))
    
    return {'acc': [round(100 * np.mean(corrects[i]),2) for i in range(3)],
            'parses': [round(100 * np.mean(parses[i]),2) for i in range(3)],
            'tokens': [round(np.mean(tokens[i]), 2) for i in range(3)],
            'time': [round(np.mean(times[i]), 2) for i in range(3)]}
        
        
        
def get_avgs_gsm(path, parser):
    res = srsly.read_jsonl(path)
    times = []
    tokens = []
    corrects = []
    parses = []
    for r in res:
        times.append(r['total_time'])
        tokens.append(r['total_tokens'])
        corrects.append(r['correct'])
        parses.append(check_gsm_parsed(r['parsed_completion'], parser))
    
    print('Mean Accuracy ', np.mean(corrects))
    print('Mean Tokens ', np.mean(tokens))
    print('Mean Time ', np.mean(times))
    print('Mean Parses ', np.mean(parses))
    return {'acc': round(100 * np.mean(corrects),2),
            'parses': round(100 * np.mean(parses),2),
            'tokens': round(np.mean(tokens), 2),
            'time': round(np.mean(times), 2)
            }

def get_results(task, model_name):
    results_dir = f"./logging/{task}/cot-model={model_name}/"
    results = {}
    if task == 'gsm_symbolic':
        num_shots = [8]
        with open("./itergen/iter_syncode/iter_syncode/parsers/grammars/gsm_grammar.lark", "r") as file:
            grammar = file.read()

        # Create a Lark parser
        parser = Lark(grammar)
        for num_shot in num_shots:
            results[f'{num_shot}_shot'] = {}
            
            results[f'{num_shot}_shot']['unconstrained'] =  get_avgs_gsm(f"{results_dir}/cot=False/parsing=regex/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl", parser)
            
            #results[f'{num_shot}_shot']['unconstrained_sample'] = get_pass_k_gsm([f"./logging_samp{i}/{task}/cot-model={model_name}/cot=False/parsing=regex/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl" for i in range(3)], parser)
            
            results[f'{num_shot}_shot']['constrained'] =  get_avgs_gsm(f"{results_dir}/cot=False/parsing=regex/gsm-gsm/cot-grammar-mode=grammar_strict/{num_shot}-shot_1_samples_True.jsonl", parser)

            results[f'{num_shot}_shot']['cot_unconstrained'] =  get_avgs_gsm(f"{results_dir}/cot=True/parsing=regex/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl", parser)  
            
            #results[f'{num_shot}_shot']['cot_unconstrained_sample'] = get_pass_k_gsm([f"./logging_samp{i}/{task}/cot-model={model_name}/cot=True/parsing=regex/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl" for i in range(3)], parser)
            
            results[f'{num_shot}_shot']['adaptive'] =  get_avgs_gsm(f"{results_dir}/cot=True/parsing=regex/gsm-gsm/cot-grammar-mode=adaptive/{num_shot}-shot_1_samples_True.jsonl", parser)      
        
    if task == 'fol':
        num_shots = [2]
        for num_shot in num_shots:
            results[f'{num_shot}_shot'] = {}
            results[f'{num_shot}_shot']['cot_constrained'] =  get_avgs_fol(f"./logging/fol/cot-model=Qwen2.5-Math-7B-Instruct/cot=False/parsing=none/prover9-prover9/cot-grammar-mode=grammar_strict/2-shot_1_samples_True.jsonl")
            results[f'{num_shot}_shot']['cot_unconstrained'] =  get_avgs_fol(f"{results_dir}/cot=True/parsing=none/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl")  
            results[f'{num_shot}_shot']['cot_unconstrained_sample'] = get_pass_k_fol([f"./logging_samp{i}/{task}/cot-model={model_name}/cot=True/parsing=none/text-text/cot-grammar-mode=original/{num_shot}-shot_1_samples_True.jsonl" for i in range(3)])
            
            #results[f'{num_shot}_shot']['adaptive_grammar'] =  get_avgs_fol(f"{results_dir}/cot=True/parsing=none/prover9-prover9/cot-grammar-mode=adaptive_grammar/{num_shot}-shot_1_samples_True.jsonl")     
            results[f'{num_shot}_shot']['adaptive'] =  get_avgs_fol(f"{results_dir}/cot=True/parsing=none/prover9-prover9/cot-grammar-mode=adaptive/{num_shot}-shot_1_samples_True.jsonl")                
    
    for k, v in results.items():
        for k1, v1 in v.items():
            print(f"{k} {k1} {v1}")

if __name__ == '__main__':
    fire.Fire(get_results)