"""Prove the selected models run on this machine: load each on the GPU, time a real call, report peak VRAM.

    .venv/Scripts/python examples/check_models.py            # each model alone, then the full stack together
"""

import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from guideline_rag import Chunk, Hit, build_prompt
from guideline_rag.adapters import CrossEncoderScorer, HFPipelineLLM, NLIJudge, SentenceTransformerEmbedder

QUESTION = "Can ramipril and valsartan be prescribed together?"
PASSAGES = [
    "An ACE inhibitor and an angiotensin receptor blocker should not be combined because the combination "
    "increases hyperkalaemia and acute kidney injury without added benefit.",
    "Smoking cessation is the single most effective intervention to slow the progression of COPD.",
]
HITS = [Hit(Chunk(id=f"P{i}", text=t), 1.0) for i, t in enumerate(PASSAGES)]

EMBEDDERS = ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3"]
RERANKERS = ["BAAI/bge-reranker-v2-m3"]
LLMS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]


def gib(n):
    return n / 1024**3


def measure(label, load, call):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = load()
    load_s = time.perf_counter() - t0
    call(model)
    t1 = time.perf_counter()
    out = call(model)
    call_s = time.perf_counter() - t1
    print(f"{label:42s} load {load_s:6.1f}s  call {call_s:6.2f}s  peak VRAM {gib(torch.cuda.max_memory_allocated()):5.2f} GiB")
    return model, out


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    total = gib(torch.cuda.get_device_properties(0).total_memory)
    print(f"GPU: {torch.cuda.get_device_name(0)}  ({total:.1f} GiB)\n")

    for name in EMBEDDERS:
        m, _ = measure(f"embed  {name}", lambda: SentenceTransformerEmbedder(name), lambda m: m.encode(PASSAGES + [QUESTION]))
        del m; gc.collect()
    for name in RERANKERS:
        m, scores = measure(f"rerank {name}", lambda: CrossEncoderScorer(name), lambda m: m(QUESTION, PASSAGES))
        print(f"{'':42s} scores {[round(s, 3) for s in scores]}  (relevant passage first = {scores[0] > scores[1]})")
        del m; gc.collect()
    m, verdict = measure("nli    cross-encoder/nli-deberta-v3-base", NLIJudge,
                         lambda m: m.judge("ACE inhibitors and ARBs should not be combined.", PASSAGES[0]))
    print(f"{'':42s} verdict {verdict}  (expected 'supports')")
    del m; gc.collect()
    for name in LLMS:
        m, text = measure(f"llm    {name}", lambda: HFPipelineLLM(name, max_new_tokens=200), lambda m: m(build_prompt(QUESTION, HITS)))
        print(f"{'':42s} reply {text[:160]!r}")
        del m; gc.collect()

    print("\nFull stack resident together (bge-m3 + bge-reranker + NLI + Qwen3-4B):")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    emb = SentenceTransformerEmbedder("BAAI/bge-m3")
    rer = CrossEncoderScorer("BAAI/bge-reranker-v2-m3")
    nli = NLIJudge()
    llm = HFPipelineLLM("Qwen/Qwen3-4B-Instruct-2507", max_new_tokens=200)
    nli.judge("ACE inhibitors and ARBs should not be combined.", PASSAGES[0])
    emb.encode(PASSAGES)
    rer(QUESTION, PASSAGES)
    llm(build_prompt(QUESTION, HITS))
    devices = set(getattr(llm.generator.model, "hf_device_map", {"": str(llm.generator.model.device)}).values())
    on_gpu = all(str(d).startswith("cuda") or d == 0 for d in devices)
    free, total_b = torch.cuda.mem_get_info()
    used = gib(total_b - free)
    print(f"LLM layer devices {sorted(map(str, devices))} -> all on GPU: {on_gpu}")
    print(f"GPU memory in use {used:.2f} GiB of {total:.1f} GiB (tensor peak {gib(torch.cuda.max_memory_allocated()):.2f})"
          f" -> {'FITS' if on_gpu and used < total * 0.9 else 'TOO TIGHT'}")


if __name__ == "__main__":
    main()
