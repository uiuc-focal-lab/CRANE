
# # # # #Qwen/Qwen2.5-1.5B-Instruct
python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "<<" --end_symbol ">>" --enable_dist True --num_workers_per_gpu 2 --num_gpus 2


# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 2 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 3 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 3 --num_gpus 2




# # # #Qwen/Qwen2.5-Math-7B-Instruct
# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2





# # #deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

# python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar4" --start_symbol "||" --end_symbol "||" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2








# # # #Qwen/Qwen2.5-1.5B-Instruct
python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 2 --num_gpus 2


python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~~" --enable_dist True --num_workers_per_gpu 2 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 3 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-1.5B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 3 --num_gpus 2




# # #Qwen/Qwen2.5-Math-7B-Instruct
python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "Qwen/Qwen2.5-Math-7B-Instruct" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2





# #deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2


python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2

python main.py --dataset gsm_symbolic  --regex_parser True --num_examples 100 --num_shots 8  --do_cot True --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --log_dir "logging_symbol_ablate_dollar5" --start_symbol "~~" --end_symbol "~~" --enable_dist True --num_workers_per_gpu 1 --num_gpus 2








