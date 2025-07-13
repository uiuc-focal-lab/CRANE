import time 

def generate_fol_with_itergen(iter_gen, problem: dict, prompt_key: str, max_iter: int, backwards_limit: int):
    start_time = time.time()

    iter_gen.start(problem[prompt_key], reasoning = problem.get('reasoning', None))
    out = iter_gen.forward(num=1)
    print(out)
    iter_gen._metadata['time'] = time.time() - start_time
    iter_gen._metadata['tokens'] = iter_gen._metadata.pop('total_tokens')
    print(iter_gen._metadata)
    return out, iter_gen._metadata