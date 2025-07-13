
# ### Qwen2.5-1.5B-Instruct

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>" 

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"


### deepseek-ai/DeepSeek-R1-Distill-Llama-8B

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 4 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"





# ### Qwen2.5-Coder-7B-Instruct

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Coder-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Coder-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Coder-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Coder-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"


# ### meta-llama/Llama-3.1-8B-Instruct

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "meta-llama/Llama-3.1-8B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"


# ### Qwen2.5-Math-7B-Instruct

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"

python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --enable_dist True --num_workers_per_gpu 1 --num_gpus 2  --start_symbol "<<" --end_symbol ">>"



