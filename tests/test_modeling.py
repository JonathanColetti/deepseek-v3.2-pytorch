# coding=utf-8
"""End2end tests for the DeepSeek V3.2 model."""

import torch

from deepseek_v3_2 import DeepSeekV32ForCausalLM, DeepSeekV32Model


def test_forward_logits_shape(tiny_model, sample_inputs):
    """ForCausalLM forward yields logits of shape (B, S, vocab)."""
    out = tiny_model(**sample_inputs)
    assert out.logits.shape == (2, 32, 1000)


def test_loss_computation(tiny_model, sample_inputs):
    """Providing labels yields a finite scalar LM loss."""
    out = tiny_model(input_ids=sample_inputs["input_ids"], labels=sample_inputs["input_ids"])
    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss)


def test_collect_dsa_losses(tiny_model, sample_inputs):
    """collect_dsa_losses returns a finite aggregated KL loss."""
    out = tiny_model(**sample_inputs, collect_dsa_losses=True)
    assert out.dsa_kl_loss is not None
    assert torch.isfinite(out.dsa_kl_loss)


def test_backbone_output(tiny_config):
    """The backbone returns last_hidden_state of the right shape."""
    model = DeepSeekV32Model(tiny_config).eval()
    x = torch.randint(0, 1000, (2, 16))
    out = model(input_ids=x)
    assert out.last_hidden_state.shape == (2, 16, tiny_config.hidden_size)


def test_generate(tiny_model, sample_inputs):
    """Greedy generation appends max_new_tokens to the prompt."""
    gen = tiny_model.generate(sample_inputs["input_ids"], max_new_tokens=5)
    assert gen.shape == (2, 37)


def test_moe_and_dense_layers(tiny_config):
    """First-k layers are dense MLP, later layers are MoE."""
    model = DeepSeekV32Model(tiny_config)
    from deepseek_v3_2.model import DeepSeekV32MLP, DeepSeekV32MoE

    assert isinstance(model.layers[0].mlp, DeepSeekV32MLP)
    assert isinstance(model.layers[1].mlp, DeepSeekV32MoE)


def test_attention_mask_padding(tiny_model):
    """Padding tokens (mask=0) do not crash the forward pass."""
    x = torch.randint(0, 1000, (2, 20))
    mask = torch.ones(2, 20, dtype=torch.long)
    mask[:, 15:] = 0
    out = tiny_model(input_ids=x, attention_mask=mask)
    assert torch.isfinite(out.logits).all()
