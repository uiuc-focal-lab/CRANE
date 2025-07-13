
import time 

def get_valid_vars(problem):
    valid_vars = set()
    for var, var_type in eval(problem['variable_types']).items():
        if var_type != 'str':
            valid_vars.add(var)
    return valid_vars

def generate_gsm_symbolic_with_itergen(iter_gen, problem: dict, prompt_key: str, max_iter: int, backwards_limit: int):
    start_time = time.time()
    valid_vars = get_valid_vars(problem)
    
    all_end_ops = [' + ', ' - ', ' * ', ' / ',  ' // ', '>>', '>> ']
    # all_end_ops += [f'{eo} ' for eo in all_end_ops]
    # all_end_ops += [f' {eo}' for eo in all_end_ops]
    # all_end_ops += [f' {eo} ' for eo in all_end_ops]
    all_end_ops.append('')
    
    all_trie_vars = [f'{var}{e_op}' for var in valid_vars for e_op in all_end_ops]
    
    iter_gen.start(problem[prompt_key], reasoning = problem.get('reasoning', None), valid_vars = all_trie_vars, valid_start_ops = [''], 
                   valid_end_ops = all_end_ops)
    #num_backwards = 0
    iter = 0
    print(valid_vars)
    is_backtrack = False
    while not iter_gen.finished() and iter < max_iter:
        iter += 1
        #import pdb; pdb.set_trace()
        out = iter_gen.forward(units=['VARIABLE'], num=1, is_backtrack=is_backtrack)
        # import pdb; pdb.set_trace()
        var_names = iter_gen.view('VARIABLE')[0]
        last_var = var_names[-1] if var_names else None        
        

        if last_var !=None and not last_var in valid_vars:
            if iter_gen.num_backwards < backwards_limit:
                iter_gen.backward('VARIABLE')
                #num_backwards += 1
                is_backtrack = False
                continue
            else:
                is_backtrack = False
                iter_gen.reset_backwards()
                
        else:
            is_backtrack = False
  
        print(out)
    iter_gen._metadata['time'] = time.time() - start_time
    iter_gen._metadata['tokens'] = iter_gen._metadata.pop('total_tokens')
    print(iter_gen._metadata)
    return out, iter_gen._metadata





def get_valid_vars_gsm_grammar(valid_vars):
    
    var_production = ' | '.join([f'"{var}"' for var in valid_vars])
    
    return f"""start: space? "<" "<" space? expr space? ">" ">" space?

expr: expr space? "+" space? term   
     | expr space? "-" space? term   
     | term

term: term space? "*" space? factor 
     | term space? "/" space? factor 
     | term space? "//" space? factor 
     | term space? "%" space? factor  
     | factor space?

factor: "-" space? factor    
       | TYPE "(" space? expr space? ")" 
       | primary space?

primary: NUMBER        
        | var      
        | "(" space? expr space? ")"

var: {var_production}

TYPE.4: "int"

space: " "

%import common.CNAME -> VARIABLE
%import common.NUMBER"""


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_terminal = False

def get_all_toks(word, root, tokenizer, valid_start_ops, valid_end_ops):
    for i in range(len(word)):
        node = root
        prefix = tokenizer.encode(word[:i + 1])
        if word[i+1:].strip() == '>':
            continue
        
        
        suffix = tokenizer.encode(word[i + 1:])
        for id in prefix + suffix:
            if id not in node.children:
                node.children[id] = TrieNode()
            node = node.children[id]

def get_valid_vars_mask(node, scores, token_ids):
    mask = torch.zeros_like(scores[0], dtype= bool, device=scores.device)
    last_token = token_ids[-1]
    if last_token in node.children:
        for child in node.children[last_token].children:
            mask[child] = 1
    else:
        for child in node.children:
            mask[child] = 1
    return mask

       


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
        if end_in_grammar:
            self.tokenized_end_symbol = torch.tensor([self.tokenizer.convert_tokens_to_ids(self.end_symbol)], device=self.device)
            if len(self.tokenized_end_symbol) > 1:
                self.tokenized_end_symbol = None
    
    def reload_grammar(self, valid_vars):
        grammar = Grammar(get_valid_vars_gsm_grammar(valid_vars))
        self._ignore_whitespace = self._get_ignore_whitespace(self.grammar)
        self.inc_parsers = [create_parser(grammar, ignore_whitespace=self._ignore_whitespace)]
        self.dfa_mask_store = DFAMaskStore.load_dfa_mask_store(
                                    grammar=grammar, 
                                    tokenizer=self.tokenizer, 
                                    use_cache=True,
                                    mode='grammar_strict', # This is default under-approximation mode in SynCode
                                    )
    
    def reset_backwards(self):
        self.num_backwards = 0
        self.current_state = UnconstrainedMode
        self.last_constrained_end = len(self.session_tokens[0]) - 1

    def start(self, prompt: Union[str, list], reasoning = None, valid_vars = None, valid_start_ops = None, valid_end_ops = None):
        """
        Start the iteration process.
        """
        self.num_backwards = 0
        self.vars_root = TrieNode()
        self.vars_cur_node = self.vars_root
        for var in valid_vars:
            get_all_toks(var, self.vars_root, self.tokenizer, valid_start_ops, valid_end_ops)
        
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
        stopping_criteria = StoppingCriteriaList()
        self.stopping_criteria = self.model._get_stopping_criteria(
            generation_config=self.generation_config, 
            stopping_criteria=stopping_criteria, 
            tokenizer=self.tokenizer 
        )

    
    @torch.inference_mode()
    def forward(self, unit:Optional[str]=None, units:Optional[Iterator[str]]=None, num:int=1, is_backtrack = False, **gen_args: dict) -> str:
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
        self.vars_cur_node = self.vars_root
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
                
                if self.end_symbol in structured_gen:
                        #import pdb; pdb.set_trace()
                        self.current_state = UnconstrainedMode
                        self.last_constrained_end = len(self.session_tokens[0]) - 1
                        self.structured_gen = ['' for _ in range(self.num_outputs)]
                        for idx, ip in enumerate(self.inc_parsers):
                            ip.reset()
                        
                        parse_results = [ip.get_acceptable_next_terminals(self.structured_gen[idx]) for idx, ip in enumerate(self.inc_parsers)]
                        initial_char_counts = [len(self.structured_gen[idx]) for idx in range(self.num_outputs)]

            next_tokens, next_token_probs = self._get_next_token_grammar(gen_mode, next_token_scores, parse_results, is_bactrack= is_backtrack)

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
                    # import pdb; pdb.set_trace()
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
            parse_results: ParseResult, 
            is_bactrack: bool = False
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
        #import pdb; pdb.set_trace()
        invalid_at_least_onnce = False
        for idx in range(self.num_outputs):
            is_valid, next_tok_idx = self._is_valid(idx, self.session_tokens[idx][self.start_constrained_from:], next_token[idx])
            next_token[idx] = next_tok_idx
            if not is_valid:
                invalid_at_least_onnce = True
                mask = self.dfa_mask_store.get_accept_mask(parse_results[idx]) 
                next_token_scores = self._apply_mask(idx, mask, next_token_scores)
            if is_bactrack:
                # import pdb; pdb.set_trace()
                invalid_at_least_onnce = True
                mask = self.dfa_mask_store.get_accept_mask(parse_results[idx]) 
                next_token_scores = self._apply_mask(idx, mask, next_token_scores)
                #self.tokenizer.batch_decode(torch.where(vars_mask == 1)[0])
                vars_mask = get_valid_vars_mask(self.vars_cur_node, next_token_scores, self.session_tokens[idx])
                # import pdb; pdb.set_trace()
                next_token_scores = self._apply_mask(idx, vars_mask, next_token_scores)

        if invalid_at_least_onnce:
            # Update the next token if the grammar is violated at least once
            next_token, next_token_probs = self._get_next_token(gen_mode, self.session_tokens, self.logit_warper, next_token_scores)

            # if '+-' in self.tokenizer.decode(torch.cat((self.session_tokens[0], next_token[0].unsqueeze(0)), dim=-1)):
            #     import pdb; pdb.set_trace()
            
            if is_bactrack:
                # import pdb; pdb.set_trace()
                try:
                    self.vars_cur_node = self.vars_cur_node.children[next_token[0].item()]
                    # import pdb; pdb.set_trace()
                except:
                    #import pdb; pdb.set_trace()
                    self.vars_cur_node = self.vars_root
            else:
                self.vars_cur_node = self.vars_root
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
            return False
        
        if r.remainder_state == RemainderState.COMPLETE or r.remainder_state == RemainderState.MAYBE_COMPLETE:
            return True, next_token

        # Check if the remainder is a valid prefix for the last terminal
        out = self.dfa_mask_store.is_valid_prefix(r)
        
        if not out:
            if self.tokenizer.decode(next_token).rstrip(',. ') == self.end_symbol and self.tokenized_end_symbol is not None:
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