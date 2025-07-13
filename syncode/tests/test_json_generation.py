import unittest
import sys
import os
import re
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import syncode.common as common
from syncode.infer import Syncode
from mxeval.data import get_data
from syncode.dataset import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm 
from collections import defaultdict
import time
from syncode.evaluation.json_eval import *

class TestSyncode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming Syncode and get_data are set up correctly in your environment
        cls.syncode = Syncode(model="test-instruct", mode='grammar_mask', device='cpu', do_sample=False, max_new_tokens=400, dataset='input', grammar='json')
        cls.problems = Dataset('json_eval').problems
        cls.tokenizer = AutoTokenizer.from_pretrained("rahuldshetty/tiny-starcoder-instruct")

    def test_syntax_json(self):
        syncode = self.syncode
        problems = self.problems
        if syncode.grammar_decoder is not None:
            syncode.grammar_decoder.parse_output_only = True

        samples = []
        outputs = []
        pbar = tqdm(total=len(problems) * syncode.num_samples)
        results = defaultdict(list)
        syntax_results = defaultdict(list)
        time1 = time.time()
        for task_id, problem in enumerate(problems):
            output = JSONEval.run_eval_for_task(syncode, syncode.num_samples, problem, samples, pbar, task_id)
            outputs.append(outputs) 

        avg_time = (time.time() - time1) / len(problems)
        syncode.logger.log_time(f"Averge time taken for each task: {avg_time:.2f}s")
        
        validate_json_data(syncode, samples, syntax_results, results)
        
        schema_pass_at_k = compute_pass_at_k(results)
        syntax_pass_at_k = compute_pass_at_k(syntax_results)
        syncode.logger.close()

        
        

