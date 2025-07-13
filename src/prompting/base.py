from jinja2 import Environment, FileSystemLoader
import yaml
import json 
from abc import ABC, abstractmethod
from textwrap import dedent
import os
from copy import deepcopy

class BasePrompter(ABC):
    def __init__(self, dataset, do_cot = True, instruct_type = 'text', num_shots=8) -> None:
        self.num_shots = num_shots
        with open(f'prompt_templates/{dataset}.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        self.fewshots = []
        if 'fewshots' in self.config:
            if do_cot:
                if 'cot' in self.config['fewshots'] and instruct_type in self.config['fewshots']['cot']:
                    self.fewshots = self.config['fewshots']['cot'][instruct_type]
            else:
                if 'std' in self.config['fewshots'] and instruct_type in self.config['fewshots']['std']:
                    self.fewshots = self.config['fewshots']['std'][instruct_type]

        self.task_specification = self.config.get('task_specification', None)
        self.task_specification = self.config.get('task_specification', None)
        self.format_instruct = None
        self.prompt_str = "[[QUESTION]]"
        if do_cot:
            if 'cot_instruct' in self.config and instruct_type in self.config['cot_instruct']:
                self.format_instruct = self.config['cot_instruct'][instruct_type]
            
            if 'cot_prompt' in self.config and  instruct_type in self.config['cot_prompt']:
                self.prompt_str = self.config['cot_prompt'][instruct_type]
            
        else:
            if 'std_instruct' in self.config and instruct_type in self.config['std_instruct']:
                self.format_instruct = self.config['std_instruct'][instruct_type]
            
            if 'std_prompt' in self.config and  instruct_type in self.config['std_prompt']:
                self.prompt_str = self.config['std_prompt'][instruct_type]
                
    def prompt(self, row, modify_system_prompt = True, chat_mode = True):
        task_spec = "" 
        if self.task_specification is not None:
            task_spec += self.task_specification 
        if self.format_instruct:
            task_spec += '\n' + self.format_instruct
        
        messages = []
        if modify_system_prompt and chat_mode:
            if task_spec != "":
                messages = [
                {"role": "system",
                "content": dedent(task_spec.strip())}]
            for example in self.fewshots[:self.num_shots]:
                messages.append(
                    {
                        "role": "user",
                        "content": example['question']
                    }
                )
                messages.append(
                {
                    "role": "assistant",
                    "content": example['response']        
                })
            
            messages.append(
            {
                "role": "user",
                "content": self.prompt_str.replace('[[QUESTION]]', row['question'])
            })
            
            messages.append(
            {
                "role": "assistant",
                "content": ""
            }) 
            return messages
        
        else:
            if task_spec != "":
                messages.append(task_spec)
            prompt = '\n'.join(messages + [example['question'] + '\n' + example['response'] for example in self.fewshots[:self.num_shots]] + [self.prompt_str.replace('[[QUESTION]]', row['question'])])
            if chat_mode:
                prompt = [{'role': 'user', 'content': prompt}]
            return prompt


class BaseParser(ABC):
    def __init__(self, dataset, do_cot = True, instruct_type = 'text', reprompt_reasoning = False, lm_parser = None, regex_parser = False, num_return_sequences = 1, start_symbol = None, end_symbol = None) -> None:
        with open(f'prompt_templates/{dataset}.yaml', 'r') as file:
            self.config = yaml.safe_load(file)
        if lm_parser is not None: 
            self.parser_prompt = self.config['parser_prompt'].get(instruct_type, None)
            self.parser = lm_parser 
        else:
            self.parser_prompt, self.parser = None, None
        self.dataset = dataset
        self.reprompt_reasoning = reprompt_reasoning
        self.regex_filter = regex_parser
        self.num_return_sequences = num_return_sequences
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
    
    def remove_tmp_paths(self):
        if hasattr(self, 'tmp_res_paths'):
            for path in self.tmp_res_paths:
                os.remove(path)
    
    def parse_completion_lm(self, batch, modify_system_prompt = True, chat_mode = True):
        if modify_system_prompt and chat_mode:
            for i in range(len(batch)):
                batch[i]['parser_prompt'] = [
                        {"role": "user", "content": batch[i]['llm_response'] + '\n\n' + self.parser_prompt}, 
                        {"role": "assistant", "content": ""}]
            new_batch = self.parser(batch, prompt_key = 'parser_prompt', response_key = 'parsed_completion', info_key = 'parse_info')
        else:
            for i in range(len(batch)):
                prompt = batch[i]['llm_response'] + '\n\n' + self.parser_prompt 
                if chat_mode:
                    prompt = [{"role": "user", "content": prompt}]
                batch[i]['parser_prompt'] = prompt
            new_batch = self.parser(batch, prompt_key = 'parser_prompt', response_key = 'parsed_completion', info_key = 'parse_info')
        completions = [ex['parsed_completion'] for ex in new_batch]
        parse_infos = [ex['parse_info'] for ex in new_batch]
        return completions, parse_infos

    def seperate_reasoning_answer(self, response):
        if self.dataset == 'spider':
            response = response.split('SELECT')
            if len(response) == 1:
                return response[0], ""
            return response[0].replace('```sql', '').strip(), 'SELECT' + response[1]
        elif self.dataset == 'fol':
            response = response.split('Predicates:')
            if len(response) == 1:
                return response[0], ""
            return response[0], 'Predicates:' + response[1]
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not supported")

    def reprompt_with_reasoning(self, batch, modify_system_prompt = True, chat_mode = True):
        for i in range(len(batch)):
            reasoning, _ = self.seperate_reasoning_answer(batch[i]['llm_response'])
            batch[i]['parser_prompt'] = deepcopy(batch[i]['prompt'])
            batch[i]['reasoning'] = reasoning
        # import pdb; pdb.set_trace()
        new_batch = self.parser(batch, prompt_key = 'parser_prompt', response_key = 'parsed_completion', info_key = 'parse_info')
        
        completions = [ex['parsed_completion'] for ex in new_batch]
        parse_infos = [ex['parse_info'] for ex in new_batch]
        return completions, parse_infos    

    def parse_completion_regex(self, response):
        pass

    @abstractmethod
    def parse_answer(self, batch, modify_system_prompt = True, chat_mode = True):
        pass
    
    def _parse_answer(self, batch, modify_system_prompt, chat_mode):
        parse_infos = [{'time': 0, 'tokens': 0} for _ in range(len(batch))]
        if self.reprompt_reasoning:
            completions, parse_infos = self.reprompt_with_reasoning(batch, modify_system_prompt, chat_mode)
        elif self.parser_prompt is not None:
            completions, parse_infos = self.parse_completion_lm(batch, modify_system_prompt, chat_mode)
        elif self.regex_filter:
            completions = [self.parse_completion_regex(ex['llm_response']) for ex in batch]
        else:
            completions = [ex['llm_response'] for ex in batch]
        return completions, parse_infos