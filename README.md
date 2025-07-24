# CRANE: Reasoning with constrained LLM generation
[![arXiv](https://img.shields.io/badge/arXiv-2502.09061-red.svg)](https://arxiv.org/abs/2502.09061)

Official Implementation of [CRANE: Reasoning with constrained LLM generation](https://arxiv.org/abs/2502.09061) published at [ICML 2025](https://icml.cc/) and the [VerifAI Workshop at ICLR 2025](https://verifai-workshop.github.io/).

## Installation

Install SynCode dependency. 
```bash
# clone syncode
cd syncode/
pip install -e .
```

Install Itergen dependency
```bash
cd struct_cot/src/itergen/iter_syncode/
pip install -e .
```

Install latex2sympy. 
```bash
cd struct_cot/src/math_evaluator/latex2sympy/
pip install -e .
```

Export environment variable for `PROVER9` like the following
```bash
export PROVER9="CRANE/src/symbolic_solvers/Prover9"
```

## How to Run Evals 

Refer to the bash scripts `src/run_{task}.sh` for reproducing the main results reported in the paper. For instance, for GSM-Symbolic:
```bash
cd src/
bash run_gsm_symbolic.sh
```

Refer to the arguments in `src/main.py` for more information and to the yaml files in `src/prompting_templates` for modifying prompt style.

### Analyzing Results

After running evaluation, the result is stored in a jsonl in the folder `logging` by default. Analyze the results by running the following:
```bash
cd src/
python get_avgs.py --model_name MODEL_NAME --task TASK
```



## Citation

```bibtex
@misc{banerjee2025cranereasoningconstrainedllm,
      title={CRANE: Reasoning with constrained LLM generation}, 
      author={Debangshu Banerjee and Tarun Suresh and Shubham Ugare and Sasa Misailovic and Gagandeep Singh},
      year={2025},
      eprint={2502.09061},
      archivePrefix={arXiv},
      primaryClass={cs.PL},
      url={https://arxiv.org/abs/2502.09061}, 
}
```
