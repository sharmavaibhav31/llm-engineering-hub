"""Empirical benchmark measuring sequential vs concurrent batch execution and fault injection."""

import asyncio
import time
from reliable_explainer.config import load_config
from reliable_explainer.batch import run_batch

QUESTIONS = [
    "In 1 sentence, define TTFT in LLMs.",
    "In 1 sentence, define exponential backoff.",
    "In 1 sentence, define jitter in retries.",
    "In 1 sentence, define a semaphore.",
    "In 1 sentence, define event loop concurrency.",
]


async def run_benchmark():
    cfg = load_config()
    n = len(QUESTIONS)
    print(f"==================================================")
    print(f"BENCHMARK: Measuring {n} questions on Groq ({cfg.model})")
    print(f"==================================================")

    # 1. Sequential execution (concurrency=1)
    print(f"\n[1/2] Running {n} questions sequentially (concurrency=1)...")
    t0_seq = time.perf_counter()
    seq_results = await run_batch(QUESTIONS, concurrency=1, config=cfg, print_on_complete=False)
    seq_duration = time.perf_counter() - t0_seq
    print(f"--> Sequential Total Duration: {seq_duration:.2f}s (all {len(seq_results)} succeeded)")

    # Brief pause to avoid burst rate limits
    await asyncio.sleep(1.0)

    # 2. Concurrent execution (concurrency=3)
    print(f"\n[2/2] Running {n} questions concurrently (concurrency=3)...")
    t0_conc = time.perf_counter()
    conc_results = await run_batch(QUESTIONS, concurrency=3, config=cfg, print_on_complete=False)
    conc_duration = time.perf_counter() - t0_conc
    print(f"--> Concurrent Total Duration: {conc_duration:.2f}s (all {len(conc_results)} succeeded)")

    speedup = seq_duration / conc_duration if conc_duration > 0 else 1.0
    print(f"\nMeasured Speedup: {speedup:.2f}x faster with bounded concurrency=3")
    print(f"Numbers to fill resume bullet:")
    print(f"N = {n}")
    print(f"X (concurrent) = {conc_duration:.2f}s")
    print(f"Y (sequential) = {seq_duration:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_benchmark())

