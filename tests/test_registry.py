from __future__ import annotations

from reglu.models import get_model_spec, list_supported_models


def test_registry_exposes_supported_models():
    assert list_supported_models() == ["llama2-7b", "phi-1.5", "zephyr-7b-beta"]
    llama = get_model_spec("llama2-7b")
    assert "q_proj" in llama.default_lora_targets["all"]
    phi = get_model_spec("phi-1.5")
    assert get_model_spec("phi") is phi
    assert "dense" in phi.default_lora_targets["all"]
    assert phi.question_start_tag == "Question: "
    zephyr = get_model_spec("zephyr-7b-beta")
    assert zephyr.answer_tag == "<|assistant|>\n"
