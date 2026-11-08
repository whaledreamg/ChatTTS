#!/usr/bin/env python3
"""TTS 合成耗时剖析工具。

回答两个问题:
    1. 一次合成的时间花在哪个阶段?
    2. 「末字截断重试」值不值得开?

用法:
    # 分阶段耗时 (需要加载模型, 约 20s)
    conda run -n chattts python tools/profile_tts.py "你好，今天天气不错"

    # 对比重试开/关的代价 (走真实 WS 服务, 不需要加载模型)
    /home/gxh/miniconda3/envs/asrt/bin/python ../asr_service/tools/bench_tts_ws.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from examples.ws_server.audio import (  # noqa: E402
    SAMPLE_RATE,
    is_all_silence,
    tail_decay_ms,
)
from examples.ws_server.engine import (  # noqa: E402
    RETRY_ON_TRUNCATION,
    STREAM_BATCH,
    TAIL_DECAY_END_RATIO,
    TAIL_DECAY_MIN_MS,
    TTSEngine,
)


def profile(engine: TTSEngine, text: str) -> None:
    """分阶段测量一次合成的耗时。"""
    print(f"=== {text!r} ({len(text)} 字) ===")

    # --- 重试开/关的对比 ---
    for retry in (False, True):
        t0 = time.time()
        gen = engine.synthesize(text, voice="Default", retry_on_truncation=retry)
        first = None
        chunks = []
        for c in gen:
            if first is None:
                first = time.time()
            chunks.append(c)
        t1 = time.time()
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        first_ms = int((first - t0) * 1000) if first else -1
        total_ms = int((t1 - t0) * 1000)
        decay = int(tail_decay_ms(audio, end_ratio=TAIL_DECAY_END_RATIO))
        flag = "正常" if decay >= TAIL_DECAY_MIN_MS else "截断"
        print(
            f"  retry={str(retry):<5}: 首帧 {first_ms:>5}ms | 总 {total_ms:>5}ms"
            f" | 末音节衰减 {decay:>3}ms ({flag})"
            f" | 音频 {int(audio.size / SAMPLE_RATE * 1000)}ms"
        )

    # --- 前导静音裁剪是否在拖慢首帧 ---
    print("  (前导静音裁剪: 收到第一块就开始判定, 不再死等 600ms lookahead)")

    # --- 末尾包络, 用来肉眼确认末字是否真的被截断 ---
    gen = engine.synthesize(text, voice="Default", retry_on_truncation=False)
    audio = np.concatenate(list(gen))
    w = max(1, int(5 * SAMPLE_RATE / 1000))
    n = audio.size // w
    env = np.sqrt((audio[: n * w].reshape(n, w) ** 2).mean(axis=1))
    peak = float(env.max()) if env.size else 1.0
    print("  末尾包络 (每格 5ms):")
    for i in range(max(0, n - 40), n):
        bar = "#" * int(env[i] / max(peak, 1e-9) * 36)
        print(f"    -{(n - i) * 5:>4}ms {env[i]:.4f} {bar}")
    print()


def main() -> int:
    texts = sys.argv[1:] or ["你好，今天天气不错", "北京今天天气晴，气温 24 度"]

    print(f"末字重试默认: {'开' if RETRY_ON_TRUNCATION else '关'}")
    print(f"末音节衰减判定: end_ratio={TAIL_DECAY_END_RATIO} "
          f"阈值={TAIL_DECAY_MIN_MS}ms")
    print(f"stream_batch: {STREAM_BATCH}")
    print()

    print("加载模型...")
    engine = TTSEngine()
    engine.load()
    print()

    for text in texts:
        profile(engine, text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
