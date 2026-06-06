# coding=utf-8
"""Benchmark DSA vs dense attention wall time across sequence lengths."""

import argparse
import time

import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


def benchmark_forward(model, input_ids, n_warmup: int = 3, n_iters: int = 10) -> float:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    for _ in range(n_warmup):
        with torch.no_grad():
            model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_iters


def _build(seq_len: int, use_dsa: bool, top_k: int) -> DeepSeekV32Config:
    return DeepSeekV32Config(
        vocab_size=1000, hidden_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=8,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=16, num_experts_per_tok=2, n_group=2, topk_group=1,
        moe_intermediate_size=256, first_k_dense_replace=1,
        use_dsa=use_dsa, dsa_top_k=min(top_k, seq_len),
        max_position_embeddings=16384,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--top-k", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for seq_len in args.seq_lengths:
        cfg_dsa = _build(seq_len, True, args.top_k)
        cfg_dense = DeepSeekV32Config(**{**cfg_dsa.to_dict(), "use_dsa": False})

        model_dsa = DeepSeekV32ForCausalLM(cfg_dsa).to(device).eval()
        model_dense = DeepSeekV32ForCausalLM(cfg_dense).to(device).eval()

        x = torch.randint(0, 1000, (1, seq_len))
        t_dsa = benchmark_forward(model_dsa, x)
        t_dense = benchmark_forward(model_dense, x)

        results.append({
            "seq_len": seq_len, "dsa_ms": t_dsa * 1000, "dense_ms": t_dense * 1000,
            "speedup": t_dense / t_dsa,
        })
        print(f"seq={seq_len:5d} | DSA: {t_dsa*1000:7.1f}ms | Dense: {t_dense*1000:7.1f}ms | Speedup: {t_dense/t_dsa:.2f}x")

        del model_dsa, model_dense
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    main()
