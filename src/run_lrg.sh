python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --cot_device auto --start_symbol "<<" --end_symbol ">>"

#python main.py --dataset gsm_symbolic --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --modify_system_prompt True  --cot_grammar_mode itergen --out_grammar gsm --cot_grammar gsm --write_file True --cot_device auto --start_symbol "<<" --end_symbol ">>"



python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --modify_system_prompt True  --cot_grammar_mode original --out_grammar text --cot_grammar text --write_file True --cot_device auto --start_symbol "<<" --end_symbol ">>"

#python main.py --dataset gsm_symbolic --do_cot True --regex_parser True --num_examples 100 --num_shots 8  --overwrite_results True --cot_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --modify_system_prompt True  --cot_grammar_mode adaptive --out_grammar gsm --cot_grammar gsm --write_file True --cot_device auto --start_symbol "<<" --end_symbol ">>"
