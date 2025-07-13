import common
import torch
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from iter_syncode.parsers.grammars import Grammar
from transformers.generation.utils import GenerationMode
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from iter_syncode.dfa_mask_store import DFAMaskStore
from iter_syncode.parse_result import ParseResult, RemainderState
from iter_syncode.parsers.incremental_parser import IncrementalParser, SymbolPosMap
from transformers.cache_utils import DynamicCache
from itergen.trace import Trace
from iter_syncode.parsers import create_base_parser, create_parser

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

class IterGen:
    """
    This main class is responsible to maintain the current state of the parser and the LLM generator.

    Attributes:
    -----------
    grammar: str
        The grammar string that defines the language.
    model_id: str
        The model id that is used for generation.
    default_unit: str
        The default unit of the grammar to be used for iteration.
    device: str
        The device to run the model on.
    quantize: bool
        Whether to quantize the model.
    gen_args: dict
        The generation configuration for the model e.g. temperature, do_sample, etc. The full list of arguments can be found in the Huggingface documentation https://huggingface.co/docs/transformers/v4.43.3/en/main_classes/text_generation#transformers.GenerationConfig
        The most common ones are:
        do_sample (bool, optional, defaults to False)
        temperature (float, optional, defaults to 1.0)
        top_k (int, optional, defaults to 50)
        top_p (float, optional, defaults to 1.0)
        stop_strings (List[str], optional, defaults to None) - A list of strings to stop generation at.
        max_new_tokens (int, optional, defaults to None) - The maximum number of new tokens to generate.
    max_tokens: int
        The maximum number of tokens in a session including the prompt tokens. Default is 1000.
    structured_gen: list[str]
        The current structured generations of the model are stored as a list of strings. There are two cases:
        1. If `parse_output_only` is True, then the session_gen stores the generated output only.
        2. If `parse_output_only` is False, then the session_gen stores prompt + generated output.
    past_key_values (`DynamicCache`):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. 
            This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True.
            Currently, we only support models that use dynamic cache. Some recent models such as Gemma2 (Hybrid cache) and Mistral (sliding window cache) are not supported. Read more about kv-cache in the Huggingface documentation https://github.com/huggingface/transformers/blob/main/docs/source/en/kv_cache.md
    trace: Trace object. 

    _metadata: dict object that stores the metadata of the current session. This includes total number of tokens generated, etc.
    """
    def __init__(
            self, 
            grammar: str, 
            model_id: str, 
            default_unit: str='start', # Default unit that represents the starting symbol of the grammar
            device:str='cuda', 
            quantize:bool=True,
            max_tokens:int=8192,
            parse_output_only:bool=True,
            recurrence_penalty:float=1.0,
            **gen_args: dict
        ) -> None:

        # Points to the character position in the structured_gen
        self.cursors = []
        self.stop_strings = gen_args.pop('stop_strings', [])
        self.grammar = Grammar(grammar)
        self.default_unit = default_unit
        self.device = device
        self.structured_gen = None
        self.num_outputs = gen_args['num_return_sequences'] if 'num_return_sequences' in gen_args else 1
        self.dev_mode = True # Warnings are raised as errors in dev mode
        self.parse_output_only = parse_output_only
        self._trace = None
        self.recurrence_penalty = recurrence_penalty
        self._metadata = None

        # Load model
        self.model_id = model_id
        self.model = common.load_model(model_id, device, quantize)
        self.tokenizer = common.load_tokenizer(model_id)
        self.device = self.model.device
        # Ignore whitespace tokens
        self._ignore_whitespace = self._get_ignore_whitespace(self.grammar)

        # Load dfa mask store
        self.dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
                                    grammar=self.grammar, 
                                    tokenizer=self.tokenizer, 
                                    use_cache=True,
                                    mode='grammar_strict', # This is default under-approximation mode in SynCode
                                    )

        # Create parsers
        self.inc_parsers: Iterator[IncrementalParser] = [
            create_parser(self.grammar, ignore_whitespace=self._ignore_whitespace) for _ in range(self.num_outputs)
            ]

        self.generation_config = GenerationConfig.from_model_config(self.model.config)
        self.max_length = max_tokens
        self.update_gen_args(**gen_args)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else -1

    
    def update_gen_args(self, **gen_args: dict) -> None:
        """
        Update the generation arguments.
        """
        self.cursors = [0 for _ in range(self.num_outputs)]
        self.generation_config.update(**gen_args)
        self.logit_warper = self.model._get_logits_warper(self.generation_config, device=self.device)

    def start(self, prompt: Union[str, list], reasoning = None):
        """
        Start the iteration process.
        """
        # import pdb; pdb.set_trace()
        for idx, ip in enumerate(self.inc_parsers):
            ip.reset()

        # Free GPU memory
        torch.cuda.empty_cache()
        self.num_backwards = 0
        self.session_prompt = prompt
        self._trace = Trace()
        self._metadata = {'total_tokens': 0}

        if (isinstance(prompt, str)):
            if reasoning is not None:
                input_batch = [prompt + reasoning]
            else:
                input_batch = [prompt]
            inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.device)
        elif (isinstance(prompt, list)):
            chat_prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            if reasoning is not None:
                input_batch = [chat_prompt + reasoning]
            else:
                input_batch = [chat_prompt]
            inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.device)

        # Set the start_from index
        if self.parse_output_only:
            self.start_from = len(inputs[0])
            self.structured_gen = ['' for _ in range(self.num_outputs)]
        else:
            self.start_from = 0
            self.structured_gen = [prompt for _ in range(self.num_outputs)]
            
        self.prompt_tokens, self.prompt_attention_mask = inputs['input_ids'], inputs['attention_mask']

        # Update model kwargs
        self.model_kwargs = {'attention_mask': self.prompt_attention_mask, 'use_cache': True, 'past_key_values': DynamicCache()}

        # Expand input_ids with `num_return_sequences` additional sequences per batch
        self.session_tokens, self.model_kwargs = self._expand_inputs_for_generation(
                                input_ids=self.prompt_tokens,
                                expand_size=self.generation_config.num_return_sequences,
                                **self.model_kwargs
                            )
        
        # Initialize the stopping criteria
        self.generation_config.max_length = self.max_length
        if self.generation_config.max_new_tokens is not None:
            self.generation_config.max_length = min(self.max_length, self.generation_config.max_new_tokens+self.prompt_tokens.shape[-1])

        self.generation_config._eos_token_tensor = IterGen._tensor_or_none(self.generation_config.eos_token_id, device=self.device)
        if self.stop_strings:
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(self.tokenizer, stop_words=self.stop_strings, input_ids_cutoff=len(inputs[0]))])
        else:
            stopping_criteria = StoppingCriteriaList()
        self.stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.generation_config, 
            stopping_criteria=stopping_criteria, 
            tokenizer=self.tokenizer 
        )


    @torch.inference_mode()
    def forward(self, unit:Optional[str]=None, units:Optional[Iterator[str]]=None, num:int=1, **gen_args: dict) -> str:
        """
        Move forward by `num` number of `unit`. 

        Args:
        -----
        unit: str (optional) `unit` is the unit of the grammar to move forward by. This could be any non-terminal or terminal symbol in the grammar.
        
        num: int (optional) The number of units to move forward by.

        units: Iterator[str] (optional) `units` is a list of units to move forward by. If `units` is provided, the function moves forward by `num` number of any of the units in the list.

        The function takes either `unit` or `units` as input. 
        If both `unit` and `units` are not provided, the default unit set during IterGen initialization is used.
        """
        if unit is None and units is None: unit = self.default_unit
        assert unit is None or units is None, "Either unit or units should be provided. Not both!"

        if unit is not None:
            # Let's just use units from now on
            units = [unit]
            unit = None

        # Update the generation configuration
        self.generation_config.update(**gen_args)
        gen_mode = self._get_generation_mode(self.generation_config)

        # Initialize the parse results
        parse_results = [ip.get_acceptable_next_terminals(self.structured_gen[idx]) for idx, ip in enumerate(self.inc_parsers)]
        initial_char_counts = [len(self.structured_gen[idx]) for idx in range(self.num_outputs)]
        
        unfinished_sequences = torch.ones(self.num_outputs, dtype=torch.long, device=self.device)
        this_peer_finished = False

        while not this_peer_finished:
            try:
                if self.model_kwargs['past_key_values']: # Get the last token if kv is cached for all previous tokens
                    input_ids = self.session_tokens[..., -1].unsqueeze(-1) 
                else:
                    input_ids = self.session_tokens

                outputs = self.model(
                    input_ids, 
                    attention_mask=self.model_kwargs['attention_mask'], 
                    past_key_values=self.model_kwargs['past_key_values'])                
            except IndexError as e:  
                raise ValueError(f"The input length exceeds the context length of the model. {e}")
            
            next_token_scores = outputs.logits[:, -1, :]

            # If recurrence penalty is not 1.0, apply it to reduce the likelihood of repeating the same token
            if self.recurrence_penalty != 1.0:
                old_following_tokens = self._trace.get_next_token_ids()
                if len(old_following_tokens) > 0:
                    # Apply the recurrence penalty to tokens indexed at old_following_tokens
                    next_token_scores[0, old_following_tokens] *= self.recurrence_penalty

            # import pdb; pdb.set_trace()
            # Select the next token
            next_tokens, next_token_probs = self._get_next_token_grammar(gen_mode, next_token_scores, parse_results)

            # Update the next token
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            # Update the next session tokens
            next_session_tokens = torch.cat([self.session_tokens, next_tokens[:, None]], dim=-1)
            self._metadata['total_tokens'] += 1

            # Update the next generation 
            next_gen = self.tokenizer.batch_decode(next_session_tokens[:, self.start_from:], skip_special_tokens=True)

            # Update the trace 
            # TODO: Handle batched trace
            prev_len, next_len = len(self.structured_gen[0]), len(next_gen[0])
            self._trace.add_token(
                next_tokens[0], 
                len(self.session_tokens[0]), 
                string=next_gen[0][prev_len:next_len],
                start_char=prev_len, 
                end_char=next_len, 
                prob=next_token_probs[0]
            )
            
            # Update the parser
            for idx, ip in enumerate(self.inc_parsers):
                ## Parsing
                try: # returns the accept sequences that are currently accepted.
                    parse_results[idx] = ip.get_acceptable_next_terminals(next_gen[idx])
                except Exception as e:
                    if self.dev_mode == True:
                        raise e
                    print(f"Exception while parsing:\n {e}")
                    continue  # Skip altering the scores for this batch
            
            # Update the current generation
            self.session_tokens = next_session_tokens
            self.structured_gen = next_gen # Update the current generation

            # Update attention mask
            self.model_kwargs['attention_mask'] = torch.cat([self.model_kwargs['attention_mask'], torch.ones((self.model_kwargs['attention_mask'].size(0), 1), dtype=self.model_kwargs['attention_mask'].dtype).to(self.device)], dim=-1)

            # Stopping criterion according to the grammar
            for idx, ip in enumerate(self.inc_parsers):
                if unfinished_sequences[idx] != 0:
                    
                    # Check if the unit is generated
                    unit_generation_finished = False

                    # Find which unit is finished from units
                    for unit in units:
                        if self.inc_parsers[idx].symbol_pos_map.get_symbol_count(unit, after=initial_char_counts[idx]) >= num:
                            unit_generation_finished = True
                            finished_unit = unit
                            break

                    if unit_generation_finished:
                        # backtrack till the last character of finished_unit
                        backtrack_till_char_pos = self.inc_parsers[idx].symbol_pos_map.get_symbol_pos_end(finished_unit, idx=-1)

                        # NOTE: Instead of breaking the tokens, we use a cursor to keep track of the position in the structured_gen
                        # self._backtrack_till_char_pos(idx, backtrack_till_char_pos, keep_trace=False)
                        self.cursors[idx] = backtrack_till_char_pos
                        unfinished_sequences[idx] = 0
                        continue
                    else:
                        self.cursors[idx] = len(self.structured_gen[idx])

            # Update the unfinished sequences
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(self.session_tokens, ())
            this_peer_finished = unfinished_sequences.max() == 0     

        # Update the model kwargs at the end of the generation 
        # self._post_update_model_kwargs(**self.model_kwargs)

        output = [gen[:self.cursors[idx]] for idx, gen in enumerate(self.structured_gen)]
        return output
    
    def reset_backwards(self) -> None:
        pass

    def backward(self, unit:Optional[str]=None, num:int=1) -> str:
        """
        Move backward by num units. 

        Args:
        -----
        unit: str (optional) `unit` is the unit of the grammar to move backward by. This could be any non-terminal or terminal symbol in the grammar.
        
        num: int (optional) The number of units to move backward by. Default is 1.
        """
        if unit is None:
            unit = self.default_unit
        
        assert unit == 'token' or self.inc_parsers[0].symbol_pos_map.is_present(unit), f"Unit {unit} is not present in the generation."
        self.num_backwards += 1
        for idx in range(self.num_outputs):
            cnt_init_tokens = len(self.session_tokens[idx])
            backtrack_till_prompt = False
            target_char_pos = None
            symbol_pos_map: SymbolPosMap = self.inc_parsers[idx].symbol_pos_map

            if unit == 'token':
                if 0 <= cnt_init_tokens - num:          
                    # Find char position where (cnt_tokens - num)'th token ends
                    cnt_tokens = cnt_init_tokens - num
                    cnt_prompt_tokens = len(self.prompt_tokens[0])
                    if self.parse_output_only:
                        target_char_pos = len(self.tokenizer.decode(self.session_tokens[idx][cnt_prompt_tokens:cnt_tokens]))
                    else:
                        target_char_pos = len(self.tokenizer.decode(self.session_tokens[idx][:cnt_tokens]))
                else:
                    backtrack_till_prompt = True
            else:
                cnt_units = symbol_pos_map.get_symbol_count(unit)
                if 0 <= cnt_units-num:
                    # Find char position where (cnt_units - num)'th symbol starts
                    # self.structured_gen[idx] will be cropped as self.structured_gen[idx][:target_char_pos] 
                    target_char_pos = symbol_pos_map.get_symbol_pos_start(unit, cnt_units-num)
                else:
                    backtrack_till_prompt = True

            if backtrack_till_prompt or (self.parse_output_only == False and target_char_pos < len(self.session_prompt)):
                print(f"Warning: The target position on backtracking {target_char_pos} is less than the prompt length. Backtracking till the prompt start.")
                target_char_pos = len(self.session_prompt)

            # Backtrack till the target position
            self._backtrack_till_char_pos(idx, target_char_pos)
             
        return self.structured_gen.copy()

    def _backtrack_till_char_pos(self, idx, target_char_pos, keep_trace=True):
        """
        Backtrack till the target character position in i-th structured generation.

        Args:
        -----
        idx: (int) The index of the sequence in the batch.
        target_char_pos: (int) The target character position to backtrack to.
        """
        # Update symbol position map and remove the units that are beyond the target_char_pos
        for ip in self.inc_parsers:
            ip.symbol_pos_map.crop(target_char_pos)

        # Store the new generation and tokens
        if self.parse_output_only:
            new_gen = self.structured_gen[idx][:target_char_pos]
        else:
            new_gen = self.structured_gen[idx][len(self.session_prompt):target_char_pos]
            
        # Find the existing tokens that will be part of the new generation
        self._trace.backtrack_to_char_pos(target_char_pos, keep_trace=keep_trace)
        chars_len_in_trace = self._trace.current_token.end_char 

        # Find the token position in the session tokens to keep
        if self._trace.current_token.position == -1: 
            # This is the root token, hence we need to only keep the prompt tokens
            token_match_len = len(self.prompt_tokens[0])
        else:
            token_match_len = self._trace.current_token.position + 1
            
        # Create the new tokenization and attention mask using the old tokens and new remainder tokens
        self.session_tokens = self.session_tokens[:,:token_match_len] # TODO: this will not work for multiple outputs
        self.model_kwargs['attention_mask'] = self.model_kwargs['attention_mask'][:, :token_match_len]

        # NOTE:: This actually makes the result worse since the model is not trained to continue with retokenization at the boundary
        # Apply boundary correction
        # self._boundary_correction(idx, target_char_pos, new_gen, chars_len_in_trace, token_match_len)

        # Update the current generation
        self.structured_gen[idx] = new_gen[:chars_len_in_trace]

        # Crop the past key values inplace (to reduce memory usage)
        self.model_kwargs['past_key_values'].crop(max_length=token_match_len-1)


    def _boundary_correction(self, idx, target_char_pos, new_gen, chars_len_in_trace, token_match_len):
        """
        NOTE:: This actually makes the result worse since the model is not trained to continue with retokenization at the boundary
        """
        if chars_len_in_trace < target_char_pos:
            remainder_char_tokens = self.tokenizer(self.structured_gen[idx][chars_len_in_trace:target_char_pos], return_tensors="pt").to(self.device)
            remainder_tokenn2d = torch.ones(self.num_outputs, len(remainder_char_tokens['input_ids'][0]), dtype=torch.int64, device=self.device)*self.pad_token_id
            remainder_tokenn2d[idx] = remainder_char_tokens['input_ids'][0]
            # TODO: this will not work for multiple outputs
            self.session_tokens = torch.cat([self.session_tokens, remainder_tokenn2d], dim=-1)
            self.model_kwargs['attention_mask'] = torch.cat([self.model_kwargs['attention_mask'], remainder_char_tokens['attention_mask']], dim=-1)
            
        # Move forward with the new tokens
        cur_trace_char_pos = chars_len_in_trace
        for i in range(token_match_len, len(self.session_tokens[0])):
            token_str = self.tokenizer.decode(self.session_tokens[0][i:i+1])
            token_len = len(token_str)
            self._trace.add_token(
                    self.session_tokens[0][i], 
                    position=i, 
                    string=token_len,
                    start_char=cur_trace_char_pos, 
                    end_char=cur_trace_char_pos+token_len
                    )
            cur_trace_char_pos += token_len

        self.structured_gen[idx] = new_gen

    def view(self, unit:Optional[str]=None) -> Iterator[Iterator[str]]:
        """
        Returns the list of all units of the specified type.
        The return value is a list of lists where top-lvel list corresponds to the output index for each sequence in the batch. The inner list corresponds to each unit of the specified type.

        Example:
        If "sentence" is a unit in the grammar and the current generation is:
        self.structured_gen[0] = "My name is John. I am a software engineer. I work at Microsoft."
        Then, self.view('sentence') will return:
        [["My name is John.", "I am a software engineer.", "I work at Microsoft."]]
        """
        if unit is None:
            unit = self.default_unit

        if unit == 'token':
            raise ValueError("Viewing the current generation by token is not supported yet.")
        
        output = []
        for i in range(self.num_outputs):
            symbol_pos_map = self.inc_parsers[i].symbol_pos_map
            output_i = []
            for pos in symbol_pos_map.get_symbol_pos_all(unit):
                output_i.append(self.structured_gen[i][pos[0]:pos[1]])
            output.append(output_i)
        return output

    def finished(self) -> bool:
        """
        Returns True if the generation is finished for all sequences in the batch.
        """
        all_finished = True
        is_stopping_criteria = self.stopping_criteria(self.session_tokens, ())
        for i in range(self.num_outputs):
            all_finished = all_finished and (is_stopping_criteria[i] or self.inc_parsers[i].symbol_pos_map.get_symbol_count("start") >= 1 or self.session_tokens[i][-1] == self.tokenizer.eos_token_id)
        return all_finished

    def _get_next_token(self, gen_mode: GenerationMode, token_ids, logit_warper, next_token_scores) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Returns the next tokens and the probability of the chosen token for each sequence in the batch. The dimension of both the returned tensors is (batch_size, 1).
        """
        if gen_mode == GenerationMode.GREEDY_SEARCH:
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
        elif gen_mode == GenerationMode.SAMPLE:
            next_token_scores = logit_warper(token_ids, next_token_scores)
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_token_probs = torch.gather(probs, 1, next_token.unsqueeze(1)).squeeze(1)
        return next_token, next_token_probs
    
    def _get_next_token_grammar(
            self, 
            gen_mode: GenerationMode, 
            next_token_scores: torch.FloatTensor,
            parse_results: ParseResult
            ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Calling grammar decoder is expensive. Hence, in the opportunist mode, we call it only when
        the standard generation is syntactically incorrect
        """
        next_token, next_token_probs = self._get_next_token(gen_mode, self.session_tokens, self.logit_warper, next_token_scores)

        # For the cases when the grammar is violated, we apply the mask
        invalid_at_least_onnce = False
        for idx in range(self.num_outputs):
            is_valid = self._is_valid(idx, self.session_tokens[idx], next_token[idx])

            if not is_valid:
                invalid_at_least_onnce = True
                mask = self.dfa_mask_store.get_accept_mask(parse_results[idx]) 
                next_token_scores = self._apply_mask(idx, mask, next_token_scores)

        if invalid_at_least_onnce:
            # Update the next token if the grammar is violated at least once
            next_token, next_token_probs = self._get_next_token(gen_mode, self.session_tokens, self.logit_warper, next_token_scores)

        return next_token, next_token_probs

    def _is_valid(self, idx: int, input_ids: torch.LongTensor, next_token: torch.LongTensor) -> bool:
        """
        Check if the next token is valid according to the grammar given the input_ids.

        Args:
            idx (int): The index of the sequence in the batch.
            input_ids (torch.LongTensor): The input ids.
            next_token (torch.LongTensor): The next token.

        Returns:
            bool: True if the next token is valid, False otherwise.
        """
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
        partial_code = self.tokenizer.decode(input_ids[self.start_from:], skip_special_tokens=True)
        try:
            r = self.inc_parsers[idx].get_acceptable_next_terminals(partial_code)
        except Exception as e:
            self.logger.log(f"Exception while parsing:\n {e}")
            return False
        
        if r.remainder_state == RemainderState.COMPLETE or r.remainder_state == RemainderState.MAYBE_COMPLETE:
            return True

        # Check if the remainder is a valid prefix for the last terminal
        out = self.dfa_mask_store.is_valid_prefix(r)
        return out

    def _get_generation_mode(
        self, generation_config: GenerationConfig
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH
        return generation_mode
    

    def _post_update_model_kwargs(self, **model_kwargs: dict) -> None:
        """     
        Update the model kwargs after each generation step. 
        On every `next` call, if a certain sequence is finished we keep padding it with pad tokens. When generation for all sequences is complete, we move pad tokens to the left of the sequence and move the other tokens to the right. This is done to keep the length of all sequences same.

        Further, we also update the attention mask to attend to all tokens that are not pad tokens.

        Example:
        Let's say we have 3 sequences with the following tokens:
        seq1: [213, 123, 234, PAD, PAD, PAD]
        seq2: [213, 123, 234, 345, 123, 3222]
        seq3: [213, 123, 234, 345, 456, PAD]

        This function will update the tokens to:
        seq1: [PAD, PAD, PAD, 213, 123, 234]
        seq2: [345, 123, 3222, 213, 123, 234]
        seq3: [PAD, 213, 123, 234, 345, 456]
        """
        
        # If all tokens in the last column are pad tokens, we can crop the last column
        if torch.all(self.session_tokens[:, -1] == self.pad_token_id):
            self.session_tokens = self.session_tokens[:, :-1]
            self.model_kwargs['attention_mask'] = self.model_kwargs['attention_mask'][:, :-1]
            self.model_kwargs['past_key_values'].crop(-1)
        
        # Iterate over all sequences and move pad tokens to the left
        # for idx in range(len(self.session_tokens)):
        #     pad_mask = self.session_tokens[idx] == self.tokenizer.pad_token_id
        #     non_pad_mask = ~pad_mask
        #     self.session_tokens[idx] = torch.cat((self.session_tokens[idx][non_pad_mask], self.session_tokens[idx][pad_mask]))

        #     # Update the attention mask
        #     self.model_kwargs['attention_mask'][idx] = torch.cat((self.model_kwargs['attention_mask'][idx][non_pad_mask], self.model_kwargs['attention_mask'][idx][pad_mask]))
            
        #     # Update the past key values

        

    def _get_ignore_whitespace(self, grammar) -> bool:
        """
        Check if the grammar allows whitespace tokens to be ignored.
        """
        base_parser = create_base_parser(grammar)
        terminals = base_parser.terminals
        ignore_terminals = base_parser.ignore_tokens
        
        import regex
        ignore_whitespace = False
        for ig_name in ignore_terminals:
            for terminal in terminals:
                if terminal.name == ig_name:
                    if regex.match(terminal.pattern.to_regexp(), ' ') is not None:
                        ignore_whitespace = True # convert to boolean tensor mask. This is useful for fast union operations
        return ignore_whitespace
    
    def _apply_mask(self, idx:int, accept_mask: torch.BoolTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._debug = False

        # Mask out invalid tokens
        if torch.sum(accept_mask) != 0: 
            if len(scores[idx]) != len(accept_mask):
                # Pad accept_mask with 0 values. Since scores[i] may be longer than tokenizer vocab size, we need to pad accept_mask with 0 values
                accept_mask = torch.cat((accept_mask, torch.zeros(len(scores[idx]) - len(accept_mask), dtype=torch.bool)))
                
            scores[idx] = scores[idx].masked_fill(~accept_mask.to(scores.device), -float("inf"))
        else: # Otherwise, report the error and mask no tokens
            if self._debug:
                print(f"Warning: No acceptable tokens in the current mask. The current generation may be invalid according to the grammar.")
        return scores
    
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs

    @staticmethod
    # Convert special tokens to tensors
    def _tensor_or_none(token, device=None):
        if token is None:
            return token

        if isinstance(token, torch.Tensor):
            return token.to(device)
        return torch.tensor(token, device=device, dtype=torch.long)

class UnconstrainedMode:
    pass

class ConstrainedMode:
    pass

class AdaptiveConstrainedDecoder(IterGen):
    def __init__(self, grammar, model_id, default_unit = 'start', device = 'cuda', quantize = True, max_tokens = 8192, parse_output_only = True, recurrence_penalty = 1, start_symbol = "<<", start_in_grammar = True, end_symbol = ">>", end_in_grammar = True, **gen_args):
        super().__init__(grammar, model_id, default_unit, device, quantize, max_tokens, parse_output_only, recurrence_penalty, **gen_args)
        assert self.num_outputs == 1, "AdaptiveConstrainedDecoder currently supports only single output generation."
        assert end_in_grammar or end_symbol is None, "End symbol should be None if end_in_grammar is False."    
        self.current_state = UnconstrainedMode
        self.start_symbol = start_symbol
        self.start_in_grammar = start_in_grammar
        self.end_in_grammar = end_in_grammar
        self.end_symbol = end_symbol
        self.tokenized_end_symbol = None
        if end_in_grammar and self.end_symbol is not None:
            if self.tokenizer.convert_tokens_to_ids(self.end_symbol) is not None:
                self.tokenized_end_symbol = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.end_symbol)], device=self.device)
                if len(self.tokenized_end_symbol) > 1:
                    self.tokenized_end_symbol = None
    

    def reset_backwards(self):
        self.num_backwards = 0
        self.current_state = UnconstrainedMode
        self.last_constrained_end = len(self.session_tokens[0]) - 1

    def start(self, prompt: Union[str, list], reasoning = None):
        """
        Start the iteration process.
        """
        self.num_backwards = 0
        for idx, ip in enumerate(self.inc_parsers):
            ip.reset()

        # Free GPU memory
        torch.cuda.empty_cache()

        self.session_prompt = prompt
        self._trace = Trace()
        self._metadata = {'total_tokens': 0}

        if (isinstance(prompt, str)):
            input_batch = [prompt]
            inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.device)
        elif (isinstance(prompt, list)):
            chat_prompt = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
            input_batch = [chat_prompt]
            inputs = self.tokenizer(input_batch, return_tensors="pt").to(self.device)

        if self.parse_output_only:
            self.start_from = len(inputs[0])
            self.start_constrained_from = None
            self.structured_gen = ['' for _ in range(self.num_outputs)]
            self.overall_gen = ['' for _ in range(self.num_outputs)]
            self.last_constrained_end = self.start_from - 1
        else:
            raise NotImplementedError("AdaptiveConstrainedDecoder currently supports only parse_output_only mode.")
        
        self.structured_offset = [0 for _ in range(self.num_outputs)]
        self.prompt_tokens, self.prompt_attention_mask = inputs['input_ids'], inputs['attention_mask']
        self.current_state = UnconstrainedMode

        # Update model kwargs
        self.model_kwargs = {'attention_mask': self.prompt_attention_mask, 'use_cache': True, 'past_key_values': DynamicCache()}

        # Expand input_ids with `num_return_sequences` additional sequences per batch
        self.session_tokens, self.model_kwargs = self._expand_inputs_for_generation(
                                input_ids=self.prompt_tokens,
                                expand_size=self.generation_config.num_return_sequences,
                                **self.model_kwargs
                            )
        
        # Initialize the stopping criteria
        self.generation_config.max_length = self.max_length
        if self.generation_config.max_new_tokens is not None:
            self.generation_config.max_length = min(self.max_length, self.generation_config.max_new_tokens+self.prompt_tokens.shape[-1])

        self.generation_config._eos_token_tensor = IterGen._tensor_or_none(self.generation_config.eos_token_id, device=self.device)
        if self.stop_strings:
            stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(self.tokenizer, stop_words=self.stop_strings, input_ids_cutoff=len(inputs[0]))])
        else:
            stopping_criteria = StoppingCriteriaList()
        self.stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.generation_config, 
            stopping_criteria=stopping_criteria, 
            tokenizer=self.tokenizer 
        )
    
    @torch.inference_mode()
    def forward(self, unit:Optional[str]=None, units:Optional[Iterator[str]]=None, num:int=1, **gen_args: dict) -> str:
        """
        Move forward by `num` number of `unit`. 

        Args:
        -----
        unit: str (optional) `unit` is the unit of the grammar to move forward by. This could be any non-terminal or terminal symbol in the grammar.
        
        num: int (optional) The number of units to move forward by.

        units: Iterator[str] (optional) `units` is a list of units to move forward by. If `units` is provided, the function moves forward by `num` number of any of the units in the list.

        The function takes either `unit` or `units` as input. 
        If both `unit` and `units` are not provided, the default unit set during IterGen initialization is used.
        """
        if unit is None and units is None: unit = self.default_unit
        assert unit is None or units is None, "Either unit or units should be provided. Not both!"
        if unit is not None:
            # Let's just use units from now on
            units = [unit]
            unit = None

        # if self.structured_gen[0] == '<<':
        #     import pdb; pdb.set_trace()
        # Update the generation configuration
        self.generation_config.update(**gen_args)
        gen_mode = self._get_generation_mode(self.generation_config)

        # Initialize the parse results
        parse_results = [ip.get_acceptable_next_terminals(self.structured_gen[idx]) for idx, ip in enumerate(self.inc_parsers)]
        initial_char_counts = [len(self.structured_gen[idx]) for idx in range(self.num_outputs)]
        
        unfinished_sequences = torch.ones(self.num_outputs, dtype=torch.long, device=self.device)
        this_peer_finished = False
        # iter = 0
        
        while not this_peer_finished:
            # print(f"Iteration: {iter}")
            # iter += 1
            try:
                if self.model_kwargs['past_key_values']: # Get the last token if kv is cached for all previous tokens
                    input_ids = self.session_tokens[..., -1].unsqueeze(-1) 
                else:
                    input_ids = self.session_tokens

                outputs = self.model(
                    input_ids, 
                    attention_mask=self.model_kwargs['attention_mask'], 
                    past_key_values=self.model_kwargs['past_key_values'])                
            except IndexError as e:  
                raise ValueError(f"The input length exceeds the context length of the model. {e}")
            
            next_token_scores = outputs.logits[:, -1, :]

            # If recurrence penalty is not 1.0, apply it to reduce the likelihood of repeating the same token
            #TODO: shouldwe keep this in adaptive mode?
            if self.recurrence_penalty != 1.0:
                old_following_tokens = self._trace.get_next_token_ids()
                if len(old_following_tokens) > 0:
                    # Apply the recurrence penalty to tokens indexed at old_following_tokens
                    next_token_scores[0, old_following_tokens] *= self.recurrence_penalty

            #import pdb; pdb.set_trace()
            # Select the next token
            if self.current_state == ConstrainedMode and self.end_symbol is not None:
                structured_gen = self.structured_gen[0]
                if self.end_symbol in structured_gen[len(self.start_symbol):]:
                        #import pdb; pdb.set_trace()
                        self.current_state = UnconstrainedMode
                        self.last_constrained_end = len(self.session_tokens[0]) - 1
                        self.structured_gen = ['' for _ in range(self.num_outputs)]
                        for idx, ip in enumerate(self.inc_parsers):
                            ip.reset()
                        
                        parse_results = [ip.get_acceptable_next_terminals(self.structured_gen[idx]) for idx, ip in enumerate(self.inc_parsers)]
                        initial_char_counts = [len(self.structured_gen[idx]) for idx in range(self.num_outputs)]

            next_tokens, next_token_probs = self._get_next_token_grammar(gen_mode, next_token_scores, parse_results)

            # Update the next token
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            # Update the next session tokens
            next_session_tokens = torch.cat([self.session_tokens, next_tokens[:, None]], dim=-1)
            self._metadata['total_tokens'] += 1
            
        
            if self.current_state == ConstrainedMode:
                # import pdb; pdb.set_trace()
                # Update the next generation 
                next_gen = self.tokenizer.batch_decode(next_session_tokens[:, self.start_constrained_from:], skip_special_tokens=True)
                
                if self.start_in_grammar:
                    next_gen = [self.start_symbol + ng for ng in next_gen]
                

                # Update the trace 
                # TODO: Handle batched trace
                prev_len, next_len = len(self.structured_gen[0]), len(next_gen[0])
                self._trace.add_token(
                    next_tokens[0], 
                    len(self.session_tokens[0]), 
                    string=next_gen[0][prev_len:next_len],
                    start_char=prev_len, 
                    end_char=next_len, 
                    prob=next_token_probs[0]
                )

                # Update the parser
                for idx, ip in enumerate(self.inc_parsers):
                    ## Parsing
                    try: # returns the accept sequences that are currently accepted.
                        parse_results[idx] = ip.get_acceptable_next_terminals(next_gen[idx])
                    except Exception as e:
                        if self.dev_mode == True:
                            raise e
                        print(f"Exception while parsing:\n {e}")
                        continue  # Skip altering the scores for this batch
                
                self.structured_gen = next_gen # Update the current generation
            else:
                #self.tokenizer.decode(next_session_tokens[0, :self.last_constrained_end + 1])
                unconstrained_gen = self.tokenizer.decode(next_session_tokens[0, self.last_constrained_end + 1:], skip_special_tokens=True)
                if self.start_symbol in unconstrained_gen:
                    self.current_state = ConstrainedMode
                    self._trace = Trace()
                    self.start_constrained_from = len(next_session_tokens[0])
                    self.structured_offset = [len(self.overall_gen[i]) + 1 for i in range(self.num_outputs)]
                    if self.start_in_grammar:
                        self.structured_gen = [self.start_symbol for _ in range(self.num_outputs)]
                        self._trace.add_token(-2, len(self.session_tokens[0]), self.start_symbol, 0, len(self.start_symbol))
                        # import pdb; pdb.set_trace()
                    else:
                        self.structured_gen = ['' for _ in range(self.num_outputs)]
                    for idx, ip in enumerate(self.inc_parsers):
                        ip.reset()
                    parse_results = [ip.get_acceptable_next_terminals(self.structured_gen[idx]) for idx, ip in enumerate(self.inc_parsers)]
                
            
            # Update the current generation
            self.session_tokens = next_session_tokens
            self.overall_gen = self.tokenizer.batch_decode(self.session_tokens[:, self.start_from:], skip_special_tokens=True)
            

            # Update attention mask
            self.model_kwargs['attention_mask'] = torch.cat([self.model_kwargs['attention_mask'], torch.ones((self.model_kwargs['attention_mask'].size(0), 1), dtype=self.model_kwargs['attention_mask'].dtype).to(self.device)], dim=-1)

            if self.current_state == ConstrainedMode:
                # Stopping criterion according to the grammar
                for idx, ip in enumerate(self.inc_parsers):
                    if unfinished_sequences[idx] != 0:
                        
                        # Check if the unit is generated
                        unit_generation_finished = False
                        
                        # Find which unit is finished from units
                        for unit in units:
                            if self.inc_parsers[idx].symbol_pos_map.get_symbol_count(unit, after=initial_char_counts[idx]) >= num:
                                unit_generation_finished = True
                                finished_unit = unit
                                break

                        if unit_generation_finished:
                            # backtrack till the last character of finished_unit
                            backtrack_till_char_pos = self.inc_parsers[idx].symbol_pos_map.get_symbol_pos_end(finished_unit, idx=-1)

                            # NOTE: Instead of breaking the tokens, we use a cursor to keep track of the position in the structured_gen
                            # self._backtrack_till_char_pos(idx, backtrack_till_char_pos, keep_trace=False)
                            self.cursors[idx] = backtrack_till_char_pos + self.structured_offset[idx]
                            unfinished_sequences[idx] = 0
                            continue
                        else:
                            self.cursors[idx] = len(self.overall_gen[idx])
            else:
                self.cursors = [len(gen) for gen in self.overall_gen]


            # Update the unfinished sequences
            unfinished_sequences = unfinished_sequences & ~self.stopping_criteria(self.session_tokens, ())
            this_peer_finished = unfinished_sequences.max() == 0     

        # Update the model kwargs at the end of the generation 
        # self._post_update_model_kwargs(**self.model_kwargs)

        output = [gen[:self.cursors[idx]] for idx, gen in enumerate(self.overall_gen)]
        return output

    def _get_next_token_grammar(
            self, 
            gen_mode: GenerationMode, 
            next_token_scores: torch.FloatTensor,
            parse_results: ParseResult
            ) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Calling grammar decoder is expensive. Hence, in the opportunist mode, we call it only when
        the standard generation is syntactically incorrect
        """
        
        next_token, next_token_probs = self._get_next_token(gen_mode, self.session_tokens, self.logit_warper, next_token_scores)
        
        # if '+-' in self.tokenizer.decode(torch.cat((self.session_tokens[0], next_token[0].unsqueeze(0)), dim=-1)):
        #     import pdb; pdb.set_trace()

        if self.current_state == UnconstrainedMode:
            return next_token, next_token_probs
        
        # if '(' in self.tokenizer.decode(next_token[0]):
        #     import pdb; pdb.set_trace()
        
        # For the cases when the grammar is violated, we apply the mask
        
        invalid_at_least_onnce = False
        for idx in range(self.num_outputs):
            is_valid, next_tok_idx = self._is_valid(idx, self.session_tokens[idx][self.start_constrained_from:], next_token[idx])
            # if not is_valid:
            #     import pdb; pdb.set_trace()
            
            next_token[idx] = next_tok_idx
            if not is_valid:
                invalid_at_least_onnce = True
                mask = self.dfa_mask_store.get_accept_mask(parse_results[idx]) 
                next_token_scores = self._apply_mask(idx, mask, next_token_scores)
            
        if invalid_at_least_onnce:
            # Update the next token if the grammar is violated at least once
            next_token, next_token_probs = self._get_next_token(gen_mode, self.session_tokens, self.logit_warper, next_token_scores)

        return next_token, next_token_probs

    def _is_valid(self, idx: int, input_ids: torch.LongTensor, next_token: torch.LongTensor) -> bool:
        """
        Check if the next token is valid according to the grammar given the input_ids.

        Args:
            idx (int): The index of the sequence in the batch.
            input_ids (torch.LongTensor): The input ids.
            next_token (torch.LongTensor): The next token.

        Returns:
            bool: True if the next token is valid, False otherwise.
        """
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=-1)
        partial_code = self.tokenizer.decode(input_ids, skip_special_tokens=True)

        if self.start_in_grammar:
            partial_code = self.start_symbol + partial_code
        
        try:
            r = self.inc_parsers[idx].get_acceptable_next_terminals(partial_code)
        except Exception as e:
            self.logger.log(f"Exception while parsing:\n {e}")
            return False, self.tokenized_end_symbol
        if r.remainder_state == RemainderState.COMPLETE or r.remainder_state == RemainderState.MAYBE_COMPLETE:
            return True, next_token

        # Check if the remainder is a valid prefix for the last terminal
        out = self.dfa_mask_store.is_valid_prefix(r)
        
        if not out:
            if self.end_symbol is not None and self.tokenizer.decode(next_token).rstrip(',. \n') == self.end_symbol and self.tokenized_end_symbol is not None:
                return True, self.tokenized_end_symbol

        return out, next_token

    def _backtrack_till_char_pos(self, idx, target_char_pos, keep_trace=True):
        """
        Backtrack till the target character position in i-th structured generation.

        Args:
        -----
        idx: (int) The index of the sequence in the batch.
        target_char_pos: (int) The target character position to backtrack to.
        """
        # Update symbol position map and remove the units that are beyond the target_char_pos
        for ip in self.inc_parsers:
            ip.symbol_pos_map.crop(target_char_pos)

        # Store the new generation and tokens
        if self.parse_output_only:
            new_gen = self.structured_gen[idx][:target_char_pos]
        else:
            new_gen = self.structured_gen[idx][len(self.session_prompt):target_char_pos]
            
        # Find the existing tokens that will be part of the new generation
        # if self.structured_gen == ['<<max_n_kx>>']:
        #     self._trace.print_trace()
        #     import pdb; pdb.set_trace()
        self._trace.backtrack_to_char_pos(target_char_pos, keep_trace=keep_trace)
        chars_len_in_trace = self._trace.current_token.end_char 

        # Find the token position in the session tokens to keep
        if self._trace.current_token.position == -1: 
            # This is the root token, hence we need to only keep the prompt tokens
            token_match_len = len(self.prompt_tokens[0])
        else:
            token_match_len = self._trace.current_token.position + 1
            
        # Create the new tokenization and attention mask using the old tokens and new remainder tokens
        self.session_tokens = self.session_tokens[:,:token_match_len] # TODO: this will not work for multiple outputs
        self.model_kwargs['attention_mask'] = self.model_kwargs['attention_mask'][:, :token_match_len]

        # NOTE:: This actually makes the result worse since the model is not trained to continue with retokenization at the boundary
        # Apply boundary correction
        # self._boundary_correction(idx, target_char_pos, new_gen, chars_len_in_trace, token_match_len)

        # Update the current generation
        if chars_len_in_trace == 0:
            chars_len_in_trace = len(self.start_symbol)
        self.structured_gen[idx] = new_gen[:chars_len_in_trace]

        # Crop the past key values inplace (to reduce memory usage)
        self.model_kwargs['past_key_values'].crop(max_length=token_match_len-1)