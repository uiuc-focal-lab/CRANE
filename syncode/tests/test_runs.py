import unittest
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import subprocess
from tqdm import tqdm
import multiprocessing
import torch

#from llama_cpp import Llama, LlamaGrammar
from collections import defaultdict
from mxeval.data import write_jsonl
from syncode.evaluation.mxeval_evaluation import compute_pass_at_k
from syncode.evaluation.json_eval import validate_json_data
from syncode.dataset import Dataset
import time
import json
from outlines.models import Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import outlines.generate.cfg as cfg
import signal
from guidance import models, gen
import outlines
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

from outlines.samplers import BeamSearchSampler, GreedySampler, MultinomialSampler
from outlines.generate.api import GenerationParameters, SamplingParameters

HF_ACCESS_TOKEN = os.environ['HF_ACCESS_TOKEN'] if 'HF_ACCESS_TOKEN' in os.environ else None

HF_CACHE = os.environ['HF_CACHE'] if 'HF_CACHE' in os.environ else 'cache/'

class TestSyncode(unittest.TestCase):
    def load_mdl(self, model_name, prompt_type, use_llama = False, use_guidance = False, use_outlines = False, use_trancfg = False, llama_file = None, device = 'cuda:0'):
        #use_llama, use_guidance, use_outlines, use_trancfg = False, False, False, False
        self.alg_name = None
        self.prompt_type = prompt_type
        
        self.generation_prompt = True if 'llama' not in model_name.lower() else False
       
        # self.prompt_type = 'explicit' #either original or explicit
        # #set one of them to true here
        # use_guidance = True
        # model_name = "google/gemma-2-2b-it"
        self.model_name = model_name

        if use_llama:
            # self.llama_llm = Llama.from_pretrained(
            #     repo_id=model_name,
            #     filename= llama_file,
            #     n_gpu_layers = -1
            # )
            self.llama_llm = Llama(model_path="/data/share/models/llama_cpp/llama-chat.gguf", chat_format="chatml", n_gpu_layers= -1, verbose = True)
            self.grammar = LlamaGrammar.from_file('/home/tarun/syncode/tests/json.gbnf')
            self.alg_name = 'llama'
        else:
            self.llama_llm = None 
            self.grammar = None
        
        if use_guidance:
            self.guidance_llm = models.Transformers(model_name, cache_dir = HF_CACHE, )
            self.guidance_llm.engine.model_obj.to(torch.bfloat16).to(device)
            self.guidance_llm.engine.device = torch.device(device)
            self.dummy_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = HF_CACHE)
            self.alg_name = 'guidance'
            
        else:
            self.guidance_llm = None 
            self.dummy_tokenizer = None
        
        if use_outlines:
             self.outlines_llm = outlines.models.transformers(model_name, device = 'cuda')
             self.outlines_llm.model.to(torch.bfloat16)
             self.generator = outlines.generate.json(self.outlines_llm, '{}')
             self.alg_name = 'outlines'
            
        else:
            self.outlines_llm = None
            self.generator = None
        
        if use_trancfg:
            self.hf_llm = AutoModelForCausalLM.from_pretrained(model_name, cache_dir = HF_CACHE).to(torch.bfloat16).to(device)
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = HF_CACHE)
            if self.hf_tokenizer.pad_token_id is None:
                self.hf_tokenizer.pad_token_id = self.hf_tokenizer.eos_token_id
            
            with open("/home/tarun/syncode/tests/json.ebnf", "r") as file:
                self.grammar = file.read()

            self.alg_name = 'transformers-cfg'

            
        else:
            self.hf_llm, self.hf_tokenizer = None, None
            

        self.problems = Dataset('json_eval').problems
    


    def test_runtime(self):
        
        #[{'use_llama': True, 'llama_file': 'Llama-3.2-1B-Instruct-Q5_K_M.gguf'}]
        
        for seed in range(1, 3):
            for kwarg in [{'use_outlines': True}]:
                #for model_name in ['google/gemma-2-2b-it']:
                for model_name in ['meta-llama/Llama-2-7b-chat-hf']:
                    for prompt_type in ['original', 'explicit']:
                        for device in ['cuda:0']:

                            
                                self.load_mdl(model_name, prompt_type, device = device, **kwarg)
                                samples = []
                                outputs = []
                                
                                pbar = tqdm(total=len(self.problems) * 1)
                                results = defaultdict(list)
                                
                                fpath = f'/home/tarun/tmp_{self.model_name.split("/")[-1]}_{self.alg_name}_{self.prompt_type}_{seed}.json'
                                if os.path.exists(fpath):
                                    continue
                                time_diff = 0
                                for task_id, problem in enumerate(self.problems):
                                    # if self.outlines_llm is not None:
                                    #     self.generator = cfg(self.outlines_llm, outlines.grammars.json) 
                                    if self.outlines_llm is not None:
                                        output = run_for_task(self.guidance_llm, self.llama_llm, self.outlines_llm, self.generator, self.hf_llm, self.hf_tokenizer, self.grammar, self.dummy_tokenizer, self.prompt_type, 1, problem, samples, pbar, task_id, self.generation_prompt)
                                    
                                    time1 = time.time() 
                                    output = run_for_task(self.guidance_llm, self.llama_llm, self.outlines_llm, self.generator, self.hf_llm, self.hf_tokenizer, self.grammar, self.dummy_tokenizer, self.prompt_type, 1, problem, samples, pbar, task_id, self.generation_prompt)
                                    time_diff += (time.time() - time1)
                                    outputs.append(output) 
                                
                                avg_time = (time_diff) / len(self.problems)
                                print(f"Averge time taken for each task: {avg_time:.2f}s")
                                
                                #correctness_result = validate_json_data(syncode = None, samples = samples, results = results)
                                #correctness_result.append({'avg_time': avg_time})
                                correctness_result = [{'avg_time': avg_time}]
                                
                                write_jsonl(fpath, correctness_result)
                                
                                # pass_at_k = compute_pass_at_k(results)
                                # print(f"Result: {pass_at_k}")

        


        

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")



def run_for_task(guidance_llm, llama_llm, outlines_llm, generator, hf_llm, hf_tokenizer, grammar, dummy_tokenizer, prompt_type, num_samples_per_task, problem, samples, pbar, task_id, generation_prompt):
    end, num_toks = None, None
    if guidance_llm is not None:
        if prompt_type == 'original':
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']}\nJSON:\n"
        else:
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']} Only output JSON.\nJSON:\n"

        if dummy_tokenizer.chat_template is not None:
            prompt = dummy_tokenizer.apply_chat_template(problem["prompt"], tokenize = False, add_generation_prompt = generation_prompt)
        else:
            prompt = problem["prompt"][0]['content']
            
        prompt = f"{prompt}```json"
        
        start = time.time()
        lm2 = guidance_llm + prompt + gen("generation", stop='```', max_tokens = 400, temperature= 0.0)
        end = time.time() - start
        batch_completions = [lm2['generation']]
        num_toks = len(dummy_tokenizer(batch_completions[0])['input_ids'])

    elif llama_llm is not None:
        try:
            if prompt_type == 'original':
                problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']}\nJSON:\n"
            else:
                problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']} Only output JSON.\nJSON:\n"
            raw_completions = [llama_llm.create_chat_completion(
                messages=[
                    problem['prompt'][0]
                ],
                response_format={
                    "type": "json_object",
                },
                temperature=0.0,
                max_tokens = 400, 
                grammar = grammar
            )]
            batch_completions = []
            for completion in raw_completions:
                msg = completion['choices'][0]['message']['content']
                batch_completions.append(msg)
        except:
            batch_completions = [""]

    elif outlines_llm is not None:
        
        if prompt_type == 'original':
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']}\nJSON:\n"
        else:
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']} Only output JSON.\nJSON:\n"
        
        
        #signal.signal(signal.SIGALRM, timeout_handler)
        
        
        prompt = outlines_llm.tokenizer.tokenizer.apply_chat_template(problem["prompt"], tokenize = False, add_generation_prompt = generation_prompt)
        #signal.alarm(180)
        #completion = outlines_llm.generate(prompt, GenerationParameters(None, None, None), None, SamplingParameters(GreedySampler()))
        try:
            completion = generator(prompt, max_tokens=400)
            #signal.alarm(0)
        except:
            torch.cuda.empty_cache()
            completion = ""
        
        batch_completions = [str(completion)]
    
    elif hf_llm is not None:
        hf_logits_processor = GrammarConstrainedLogitsProcessor(IncrementalGrammarConstraint(grammar, "root", hf_tokenizer))
        if prompt_type == 'original':
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']}\nJSON:\n"
        else:
            problem['prompt'][0]['content'] = f"{problem['prompt'][0]['content']} Only output JSON.\nJSON:\n"
            
        
        if hf_tokenizer.chat_template is not None:
            prompt = hf_tokenizer.apply_chat_template(problem["prompt"], tokenize = False, add_generation_prompt = generation_prompt)
        else:
            prompt = problem["prompt"][0]['content']
        
        input_ids = hf_tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                hf_llm.device
            )  # Move input_ids to the same device as model

        output = hf_llm.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=400,
            logits_processor=[hf_logits_processor],
            num_return_sequences=1,
        )
        # decode output
        completion = hf_tokenizer.decode(output[0][len(input_ids[0]):], skip_special_tokens=True)
        batch_completions = [completion]
    
    print(batch_completions[0])
    
    for completion_id, completion in enumerate(batch_completions):
        result = dict(
                task_id = task_id, 
                completion_id = completion_id,
                completion=completion, 
                ground_truth = problem["ground_truth"], 
                schema = problem["schema"], 
                ex_time = end, 
                toks = num_toks
            )
        samples += [result]
    pbar.update(num_samples_per_task)
    return batch_completions
        




