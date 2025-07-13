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

    iter_gen.start(problem[prompt_key], reasoning = problem.get('reasoning', None))
    iter = 0
    print(valid_vars)
    while not iter_gen.finished() and iter < max_iter:
        iter += 1
        out = iter_gen.forward(units=['VARIABLE'], num=1)
        var_names = iter_gen.view('VARIABLE')[0]
        last_var = var_names[-1] if var_names else None        

        if last_var !=None and not last_var in valid_vars:
            if iter_gen.num_backwards < backwards_limit:
                iter_gen.backward('VARIABLE')
                continue
            else:
                iter_gen.reset_backwards()

        print(out)
    iter_gen._metadata['time'] = time.time() - start_time
    iter_gen._metadata['tokens'] = iter_gen._metadata.pop('total_tokens')
    print(iter_gen._metadata)
    return out, iter_gen._metadata