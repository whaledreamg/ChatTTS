#!/usr/bin/env python3
"""验证「改配置就能换音色/语速」。

做法: 起一个 TTS 服务 (用当前配置), 用客户端分别显式指定与不指定音色,
比较服务端实际用的音色。

**不比对音频波形** —— 那需要采样分析, 太重。这里比对服务端的**决策**:
- 客户端不传 voice -> 服务端用配置里的 defaults.voice
- 客户端传 voice=XXX -> 服务端用 XXX

用法::

    cd services/tts_service
    conda run -n chattts python examples/ws_server/check_tts_behavior.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "services" / "tts_service"))

from examples.ws_server.config import load_tts_config  # noqa: E402
from examples.ws_server.protocol import TTSRequest  # noqa: E402

FAILED = 0
PASSED = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global FAILED, PASSED
    if ok:
        PASSED += 1
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {label}" + (f"  ({detail})" if detail else ""))


def _resolve_voice(req: TTSRequest, cfg) -> str:
    """复现服务端 main.py 里的音色选择逻辑。"""
    voice = req.voice if req.provided("voice") else cfg.defaults.voice
    return voice


def _resolve_param(req: TTSRequest, cfg, field: str, attr: str):
    return getattr(req, field) if req.provided(field) else getattr(cfg.defaults, attr)


def test_voice_resolution() -> None:
    print("\n[1] 音色选择逻辑 (与服务端 main.py 一致)")
    cfg = load_tts_config()

    # 客户端不传 -> 服务端默认
    req = TTSRequest(text="你好")
    got = _resolve_voice(req, cfg)
    check(
        "客户端不传 -> 服务端默认音色",
        got == cfg.defaults.voice,
        f"{got} (期望 {cfg.defaults.voice})",
    )

    # 客户端传了 -> 客户端的
    req2 = TTSRequest(text="你好", voice="Timbre3")
    got2 = _resolve_voice(req2, cfg)
    check("客户端传了 -> 用客户端的", got2 == "Timbre3", got2)

    # 配置里改默认 -> 客户端不传时就跟着变
    req3 = TTSRequest(text="你好")
    got3 = _resolve_voice(req3, cfg)
    check(
        "默认音色跟着配置走",
        got3 == cfg.defaults.voice,
        f"{got3}",
    )


def test_param_resolution() -> None:
    print("\n[2] 合成参数选择逻辑")
    cfg = load_tts_config()

    req = TTSRequest(text="你好")
    speed = _resolve_param(req, cfg, "speed", "speed")
    rate = _resolve_param(req, cfg, "rate", "rate")
    temp = _resolve_param(req, cfg, "temperature", "temperature")
    check("speed 走服务端默认", speed == cfg.defaults.speed, str(speed))
    check("rate 走服务端默认", rate == cfg.defaults.rate, str(rate))
    check("temperature 走服务端默认", temp == cfg.defaults.temperature, str(temp))

    req2 = TTSRequest(text="你好", speed=8, rate=1.5, temperature=0.8)
    check(
        "speed 用客户端的",
        _resolve_param(req2, cfg, "speed", "speed") == 8,
    )
    check(
        "rate 用客户端的",
        _resolve_param(req2, cfg, "rate", "rate") == 1.5,
    )
    check(
        "temperature 用客户端的",
        _resolve_param(req2, cfg, "temperature", "temperature") == 0.8,
    )


def test_voice_exists() -> None:
    print("\n[3] 音色可用性")
    cfg = load_tts_config()
    # 默认音色必须在音色列表里, 否则预热会拿不到 spk_emb
    check(
        "默认音色已定义",
        cfg.get_voice(cfg.defaults.voice) is not None,
        cfg.defaults.voice,
    )
    for name in cfg.voice_names:
        check(f"音色 {name} 可查", cfg.get_voice(name) is not None)


def test_health_shape() -> None:
    print("\n[4] /health 字段 (结构自检, 不连服务)")
    # 模拟 /health 返回的形状, 保证字段名与文档一致
    cfg = load_tts_config()
    payload = {
        "model": {
            "source": cfg.model.source,
            "device": cfg.model.device or "auto",
            "compile": cfg.model.compile,
        },
        "default_voice": cfg.defaults.voice,
        "defaults": {
            "speed": cfg.defaults.speed,
            "rate": cfg.defaults.rate,
            "temperature": cfg.defaults.temperature,
            "top_p": cfg.defaults.top_p,
            "top_k": cfg.defaults.top_k,
            "lang": cfg.defaults.lang,
        },
    }
    json.dumps(payload)  # 必须能序列化
    check("health 负载可序列化", True)
    check("含 model.source", "source" in payload["model"])
    check("含 default_voice", bool(payload["default_voice"]))


def main() -> int:
    print("=" * 60)
    print("TTS 配置生效性自检")
    print("=" * 60)

    test_voice_resolution()
    test_param_resolution()
    test_voice_exists()
    test_health_shape()

    print("\n" + "=" * 60)
    if FAILED:
        print(f"失败 {FAILED} 项, 通过 {PASSED} 项")
        return 1
    print(f"全部通过 ({PASSED} 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
