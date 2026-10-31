# CRANE: Reasoning with constrained LLM generation
[![arXiv](https://img.shields.io/badge/arXiv-2502.09061-red.svg)](https://arxiv.org/abs/2502.09061)

Official Implementation of [CRANE: Reasoning with constrained LLM generation](https://arxiv.org/abs/2502.09061) published at [ICML 2025](https://icml.cc/) and the [VerifAI Workshop at ICLR 2025](https://verifai-workshop.github.io/).

## Installation

```bash
python -m pip install -e .
```

CRANE, IterGen, and baseline grammar-constrained inference use one shared
`crane.decoding` backend. There are no separate SynCode or IterSynCode source
packages or editable installs. Cursor and backtracking support live in the same
incremental parser used by the baseline logits processor.

```python
from crane import CRANE, IterGen
from crane.models import BaseLM
from crane.decoding.parsers.grammars import Grammar
```

`HF_CACHE` selects the model-weight cache (default: the standard Hugging Face
hub cache). `CRANE_CACHE` selects generated parser and DFA-mask caches. The old
`ITER_SYNCODE_CACHE` and `SYNCODE_CACHE` environment variables remain accepted
as fallbacks. Generated caches use a new `decoding-v1` subdirectory because old
pickles reference the removed module names and must be rebuilt once.

For first-order logic evaluation, set `PROVER9` to the directory containing
the executable, or use the bundled binary in a source checkout.

## How to Run Evals 

Refer to the bash scripts `src/crane/run_{task}.sh` for reproducing the main results reported in the paper. For instance, for GSM-Symbolic:
```bash
bash src/crane/run_gsm_symbolic.sh
```

Use `crane-eval --help` for arguments and `src/crane/prompt_templates` for prompt styles.

### Analyzing Results

After running evaluation, the result is stored in a jsonl in the folder `logging` by default. Analyze the results by running the following:
```bash
python -m crane.get_avgs --model_name MODEL_NAME --task TASK
```



## Tests

```bash
python -m pip install pytest
python -m pytest tests -q
```

Set `CRANE_TEST_MODEL` to a cached model ID or local model path to include
the real-model baseline, cursor/backtracking, and adaptive-decoding checks.

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
