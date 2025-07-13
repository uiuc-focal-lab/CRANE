from .base import BasePrompter, BaseParser
import random
import re
from typing import Optional
from mxeval.data import write_jsonl
from tqdm import tqdm
import signal
from syncode.parsers import create_base_parser
from syncode.parsers.grammars.grammar import Grammar
import syncode.evaluation.fol_eval as fol_eval


class FOLParser(BaseParser):
    def _replace_symbols(self, completion):
        completion = completion.replace('{and}', '∧')
        completion = completion.replace('{or}', '∨')
        completion = completion.replace('{not}', '¬')
        completion = completion.replace('{xor}', '⊕')
        completion = completion.replace('{implies}', '→')
        completion = completion.replace('{iff}', '↔')
        completion = completion.replace("{forall} ", '∀')
        completion = completion.replace("{exists} ", '∃')
        completion = completion.replace("{forall}", '∀')
        completion = completion.replace("{exists}", '∃')
        return completion

    def _parse_single_completion(self, input_ex, completion, parse_info):
        if 'Predicates:' in completion:
            completion = 'Predicates:' + completion.split('Predicates:')[1]
        logic_program = completion.split('------')[0]
        
        print(f"\nLogic Program:\n {logic_program}\n\n")
        logic_program = self._replace_symbols(logic_program)
        logic_program = logic_program.split('Note')[0]
        logic_program = logic_program.strip()
        print(f"\nLogic Program after symbol replacement:\n {logic_program}\n\n")

        answer = None
        compiles = False
        rand_ans = False
        error_message = None

        try:
            self.fol_parser.parse(logic_program)
            is_parsed = True
        except:
            is_parsed = False

        try:
            # import pdb; pdb.set_trace()
            program = fol_eval.FOL_Prover9_Program(logic_program)
            if program.compiles:
                compiles = True
            else:
                raise Exception("Failed to compile logic program")
            answer, error_message = program.execute_program()
            answer = program.answer_mapping(answer)
        
        except Exception as e:
            print(e)
            error_message = str(e)

        if answer is None:
            print("\n\nRandomly choosing answer\n\n")
            answer = random.choice(['A', 'B', 'C'])
            rand_ans = True

        map_label_to_answer = {'True': 'A', 'False': 'B', 'Uncertain': 'C'}
        ground_truth = map_label_to_answer[input_ex['answer']]

        print(f"rand_ans: {rand_ans}, answer: {answer}, ground_truth: {ground_truth}")
        print(f"FINAL ANSWER: {(not rand_ans) and (answer == ground_truth)}")

        return dict(
            idx=input_ex['idx'],
            correct=((not rand_ans) and (answer == ground_truth)),
            llm_completion = input_ex['llm_response'], 
            compiles=compiles,
            is_parsed=is_parsed,
            random=(rand_ans),
            logic_program=logic_program,
            answer=answer,  
            ground_truth=ground_truth,
            error_message=error_message,
            total_time = input_ex['response_info']['time'] + parse_info['time'],
            total_tokens = input_ex['response_info']['tokens'] + parse_info['tokens'],
            resp_time = input_ex['response_info']['time'],
            resp_tokens = input_ex['response_info']['tokens'],
            parse_time = parse_info['time'],
            parse_tokens = parse_info['tokens']
        )


    def parse_answer(self, batch, modify_system_prompt = True, chat_mode = True):
        if not hasattr(self, 'fol_parser'):
            self.fol_parser = create_base_parser(Grammar('prover9'))
        
        results = []
        completions, parse_infos = None, None
        if self.parser_prompt is not None:
            completions, parse_infos = self._parse_answer(batch, modify_system_prompt, chat_mode)
        else:
            completions = [ex['llm_response'] for ex in batch]
            parse_infos = [{'time': 0, 'tokens': 0} for _ in range(len(batch))]

        for i in range(len(batch)):
            completion, parse_info = completions[i], parse_infos[i]
            results.append(self._parse_single_completion(batch[i], completion, parse_info))
        return results
