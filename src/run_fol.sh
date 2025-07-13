### Qwen/Qwen2.5-7B-Instruct

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode original --cot_grammar text --out_grammar text --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode grammar_strict --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive_grammar --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


# ### Qwen/Qwen2.5-Math-7B-Instruct

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode original --cot_grammar text --out_grammar text --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode grammar_strict --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive_grammar --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


# ### meta-llama/Llama-3.1-8B-Instruct

# python main.py --do_cot True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode original --cot_grammar text --out_grammar text --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode grammar_strict --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --dataset fol --overwrite_results True --num_shots 2 --cot_grammar_mode adaptive_grammar --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2




python main.py --do_cot True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --dataset fol --num_shots 2 --cot_grammar_mode original --cot_grammar text --out_grammar text --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n"  --log_dir "logging2"

#--enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --dataset fol --num_shots 2 --cot_grammar_mode grammar_strict --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --do_cot True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --dataset fol --num_shots 2 --cot_grammar_mode adaptive --cot_grammar prover9 --out_grammar prover9 --max_tokens 800 --cot_device 'cuda:1' --write_file True --modify_system_prompt True  --start_symbol "Predicates:\n" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

