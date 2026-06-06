# coding=utf-8
"""Benchmark peak memory of DSA vs dense attention across sequence lengths."""

import argparse

import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


def measure_peak_memory(model, input_ids) -> float:
    """Return peak allocated memory in MiB for a forward pass (CUDA only)."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if device.type != "cuda":
        # CPU fallback: estimate activation footprint as element count of logits.
        with torch.no_grad():
            out = model(input_ids)
        return out.logits.element_size() * out.logits.nelement() / (1024 ** 2)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        model(input_ids)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def _build(seq_len, use_dsa, top_k):
    return DeepSeekV32Config(
        vocab_size=1000, hidden_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=8,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=16, num_experts_per_tok=2, n_group=2, topk_group=1,
        moe_intermediate_size=256, first_k_dense_replace=1,
        use_dsa=use_dsa, dsa_top_k=min(top_k, seq_len), max_position_embeddings=16384,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for seq_len in args.seq_lengths:
        cfg_dsa = _build(seq_len, True, args.top_k)
        cfg_dense = DeepSeekV32Config(**{**cfg_dsa.to_dict(), "use_dsa": False})

        m_dsa = DeepSeekV32ForCausalLM(cfg_dsa).to(device).eval()
        m_dense = DeepSeekV32ForCausalLM(cfg_dense).to(device).eval()
        x = torch.randint(0, 1000, (1, seq_len))

        mem_dsa = measure_peak_memory(m_dsa, x)
        mem_dense = measure_peak_memory(m_dense, x)
        results.append({"seq_len": seq_len, "dsa_mib": mem_dsa, "dense_mib": mem_dense})
        print(f"seq={seq_len:5d} | DSA: {mem_dsa:8.1f} MiB | Dense: {mem_dense:8.1f} MiB")

        del m_dsa, m_dense
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
