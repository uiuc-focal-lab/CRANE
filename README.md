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
export PROVER9="/home/tarun/struct_cot/src/symbolic_solvers/Prover9"
```

## How to Run Evals 

Example:
```bash
cd src/
bash run_gsm_symbolic.sh
```

Refer to `src/prompting_templates` for modifying prompt style.



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
