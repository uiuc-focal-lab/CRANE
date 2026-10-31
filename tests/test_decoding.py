"""Public decoder contracts and optional real-model release checks."""
import importlib
import os
from pathlib import Path

import pytest


def test_shared_backend():
    from crane import main
    from crane.decoding import grammar_decoder, language_model
    from crane.decoding.parsers.grammars import Grammar
    assert main.Grammar is grammar_decoder.Grammar is language_model.Grammar is Grammar
    assert main.DFAMaskStore is grammar_decoder.DFAMaskStore
    assert main.KeywordsStoppingCriteria is language_model.KeywordsStoppingCriteria


@pytest.mark.parametrize("output", ["text", ["text"], {"name": "calculate"}])
def test_callback_context_and_output(monkeypatch, output):
    from crane.models import base_model

    class Decoder:
        def __init__(self, **kwargs):
            assert "constraint_fn" not in kwargs

    calls = []

    def callback(decoder, problem, prompt_key, iterations, backwards, **context):
        calls.append((problem, context))
        return output, {"tokens": 3}

    monkeypatch.setattr(base_model, "IterGen", Decoder)
    model = base_model.BaseLM("test", mode="itergen", constraint_fn=callback)
    result = model([{"prompt": "hello", "definitions": ["tool"]}],
                   tools_key="definitions", spec="policy", trace=[])
    assert calls == [({"prompt": "hello", "definitions": ["tool"], "tools": ["tool"]},
                      {"spec": "policy", "trace": []})]
    assert result[0]["llm_response"] == (output[0] if isinstance(output, list) else output)
    model([{"prompt": "legacy"}])
    assert calls[-1] == ({"prompt": "legacy"}, {})


def test_callback_exception_propagates(monkeypatch):
    from crane.models import base_model

    class Decoder:
        def __init__(self, **kwargs):
            pass

    def fail(*args, **kwargs):
        raise RuntimeError("callback failure")

    monkeypatch.setattr(base_model, "IterGen", Decoder)
    model = base_model.BaseLM("test", mode="itergen", constraint_fn=fail)
    with pytest.raises(RuntimeError, match="callback failure"):
        model([{"prompt": "hello"}])


@pytest.mark.parametrize("decoder_name", ["IterGen", "CRANE"])
@pytest.mark.parametrize("tools", [None, [{"type": "function", "function": {"name": "calculate"}}]])
def test_chat_template_receives_tools(decoder_name, tools):
    from crane import main
    captured = {}

    class TemplateReached(Exception):
        pass

    class Tokenizer:
        def apply_chat_template(self, prompt, **kwargs):
            captured.update(kwargs)
            raise TemplateReached

    cls = getattr(main, decoder_name)
    decoder = cls.__new__(cls)
    decoder.inc_parsers = []
    decoder.tokenizer = Tokenizer()
    with pytest.raises(TemplateReached):
        decoder.start([{"role": "user", "content": "hello"}], tools=tools)
    assert captured["tools"] == tools
    assert captured["add_generation_prompt"] is True


@pytest.mark.parametrize("language,source", [
    ("json", '{"items": [1, true, null]}'),
    ("calc", "1 + 2"),
    ("python", "def f(x):\n    return x + 1\n"),
    ("go", "package main\nfunc main() {\n    return\n}\n"),
])
def test_parser_incremental_reset(language, source, tmp_path, monkeypatch):
    from crane.decoding import common
    from crane.decoding.parsers import create_parser
    from crane.decoding.parsers.grammars import Grammar
    monkeypatch.setattr(common, "DECODING_CACHE", str(tmp_path) + "/")
    parser = create_parser(Grammar(language))
    for end in range(1, len(source) + 1):
        assert parser.get_acceptable_next_terminals(source[:end]) is not None
    parser.reset()
    assert parser.cur_pos == 0
    assert parser.symbol_pos_map.get_symbol_count("start") == 0
    assert parser.get_acceptable_next_terminals(source) is not None


def test_prover9_respects_environment(monkeypatch):
    monkeypatch.setenv("PROVER9", "/custom/prover9/bin")
    from crane.symbolic_solvers.fol_solver import prover9_solver
    importlib.reload(prover9_solver)
    assert os.environ["PROVER9"] == "/custom/prover9/bin"


def test_prover9_default_is_package_relative(monkeypatch):
    monkeypatch.delenv("PROVER9", raising=False)
    from crane.symbolic_solvers.fol_solver import prover9_solver
    importlib.reload(prover9_solver)
    expected = Path(prover9_solver.__file__).resolve().parents[1] / "Prover9/bin"
    assert Path(os.environ["PROVER9"]) == expected


@pytest.mark.parametrize("dataset", ["gsm_symbolic", "fol"])
@pytest.mark.parametrize("class_name", ["BasePrompter", "BaseParser"])
def test_prompt_templates_outside_source_directory(dataset, class_name, tmp_path, monkeypatch):
    from crane.prompting import base, PARSE_MAP
    monkeypatch.chdir(tmp_path)
    cls = base.BasePrompter if class_name == "BasePrompter" else PARSE_MAP[dataset]["text"]
    instance = cls(dataset)
    assert isinstance(instance.config, dict)
    assert instance.config


@pytest.mark.skipif(not os.environ.get("CRANE_TEST_MODEL"), reason="Set CRANE_TEST_MODEL for real-weight checks")
@pytest.mark.parametrize("mode", ["baseline", "itergen", "adaptive"])
def test_real_model_decoding(mode):
    from crane import CRANE, IterGen
    from crane.decoding.infer import Syncode
    model = os.environ["CRANE_TEST_MODEL"]
    device = os.environ.get("CRANE_TEST_DEVICE", "cpu")
    if mode == "baseline":
        decoder = Syncode(model=model, grammar='start: "42"', mode="grammar_strict",
                          device=device, max_new_tokens=8, do_sample=False)
        output = decoder.model.generate_grammar_constrained_completion("The answer is ", 1)
        assert output[0].strip() == "42"
    elif mode == "itergen":
        decoder = IterGen(grammar='start: "[" answer "]"\nanswer: "42"', model_id=model,
                          device=device, max_tokens=256, max_new_tokens=8, do_sample=False)
        decoder.start("The answer is ")
        decoder.forward(unit="answer")
        assert decoder.view("answer")[0][0].strip() == "42"
        decoder.backward(unit="answer")
        decoder.forward(unit="answer")
        assert decoder.view("answer")[0][0].strip() == "42"
    else:
        decoder = CRANE(grammar='start: "<answer>" answer "</answer>"\nanswer: "42"',
                        model_id=model, device=device, max_tokens=512, max_new_tokens=64,
                        do_sample=False, start_symbol="<answer>", end_symbol="</answer>")
        prompt = "User: Answer 42.\nAssistant: <answer>42</answer>\n" * 3
        decoder.start(prompt + "User: Answer 42.\nAssistant: ")
        assert "<answer>42</answer>" in decoder.forward(unit="start")[0]
