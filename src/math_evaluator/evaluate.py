import argparse
import numpy as np
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError

from .grader import *

from .parser import *


def evaluate(data_name, batch, pred_key = 'parsed_completion'):
    answers = [parse_ground_truth(example, data_name)[1] for example in batch]
    params = [(idx, ex[pred_key], ans) for idx, (ex, ans) in enumerate(zip(batch, answers))]

    scores = []
    timeout_cnt = 0 

    with ProcessPool(max_workers=1) as pool:
        future = pool.map(math_equal_process, params, timeout=3)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                scores.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                print(error)
                scores.append(False)
                timeout_cnt += 1
            except Exception as error:
                print(error.traceback)
                exit()
                

    
    return answers, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, prompt_type=args.prompt_type, file_path=args.file_path,
             max_num_samples=args.max_num_samples, execute=args.execute)
