# coding=utf-8
"""Benchmark generation / decode throughput (tokens per second)."""

import argparse
import time

import torch

from deepseek_v3_2 import DeepSeekV32Config, DeepSeekV32ForCausalLM


def measure_decode_throughput(model, prompt_len: int, gen_len: int, batch: int = 1) -> dict:
    device = next(model.parameters()).device
    x = torch.randint(0, 1000, (batch, prompt_len), device=device)

    # Warmup
    with torch.no_grad():
        model.generate(x, max_new_tokens=2)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=gen_len)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = (out.shape[1] - prompt_len) * batch
    return {
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "batch": batch,
        "elapsed_s": elapsed,
        "tokens_per_s": new_tokens / elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DeepSeekV32Config(
        vocab_size=1000, hidden_size=512, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=8,
        q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
        n_routed_experts=16, num_experts_per_tok=2, n_group=2, topk_group=1,
        moe_intermediate_size=256, first_k_dense_replace=1,
        use_dsa=True, dsa_top_k=256, max_position_embeddings=16384,
    )
    model = DeepSeekV32ForCausalLM(cfg).to(device).eval()
    stats = measure_decode_throughput(model, args.prompt_len, args.gen_len, args.batch)
    print(f"prompt={stats['prompt_len']} gen={stats['gen_len']} batch={stats['batch']} "
          f"| {stats['tokens_per_s']:.1f} tok/s ({stats['elapsed_s']:.2f}s)")
    return stats


if __name__ == "__main__":
    main()
