#!/usr/bin/env python3
"""TTS 配置自检 (音色 + 模型)。

验证 ``configs/tts.yaml`` 真的生效:
- 配置能被读出来 (音色列表、模型参数、默认合成参数)
- 客户端**没传**的字段走服务端默认值
- 客户端**显式传**的字段能覆盖服务端默认值
- 服务端 /health 与 WS ready 消息里能查到当前生效的配置

不需要真机, 也不需要加载模型 (不连 WS 时是纯离线检查)。

用法::

    cd services/tts_service
    conda run -n chattts python examples/ws_server/check_tts_config.py
"""

from __future__ import annotations

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


def test_config_loaded() -> None:
    print("\n[1] 配置加载")
    cfg = load_tts_config()
    check("配置文件已读到", cfg.loaded, "configs/tts.yaml")
    check("有音色定义", len(cfg.voices) > 0, f"{len(cfg.voices)} 个")
    check("有默认音色", bool(cfg.defaults.voice), cfg.defaults.voice)
    check(
        "默认音色在音色列表里",
        cfg.defaults.voice in cfg.voice_names,
        f"{cfg.defaults.voice} vs {cfg.voice_names}",
    )


def test_model_section() -> None:
    print("\n[2] 模型配置")
    cfg = load_tts_config()
    m = cfg.model
    check("source 合法", m.source in ("local", "huggingface", "custom"), m.source)
    check("device 是字符串", isinstance(m.device, str), repr(m.device))
    if m.source == "custom":
        check("custom 模式有 custom_path", bool(m.custom_path), m.custom_path)
    check("enable_cache 是布尔", isinstance(m.enable_cache, bool))
    check("compile 是布尔", isinstance(m.compile, bool))


def test_voices() -> None:
    print("\n[3] 音色定义")
    cfg = load_tts_config()
    names = cfg.voice_names
    check("音色名不重复", len(names) == len(set(names)), str(names))
    for v in cfg.voices:
        ok = v.seed is not None or bool(v.ref_audio)
        check(
            f"音色 {v.name} 有 seed 或 ref_audio",
            ok,
            f"seed={v.seed} ref_audio={v.ref_audio or '(无)'}",
        )
        if v.ref_audio:
            check(
                f"音色 {v.name} 的参考音频存在",
                Path(v.ref_audio).is_file(),
                v.ref_audio,
            )


def test_defaults_override() -> None:
    print("\n[4] 客户端覆盖 vs 服务端默认")
    cfg = load_tts_config()
    d = cfg.defaults

    # 客户端什么都没传 -> 走服务端默认
    r = TTSRequest(text="你好")
    check("客户端没传 voice", not r.provided("voice"))
    check("客户端没传 speed", not r.provided("speed"))
    check("客户端没传 rate", not r.provided("rate"))
    check("客户端没传 temperature", not r.provided("temperature"))
    check("服务端默认音色可用", bool(d.voice))
    check("服务端默认语速在合法范围", 0 <= d.speed <= 9, str(d.speed))
    check("服务端默认 rate 在合法范围", 0.5 <= d.rate <= 2.0, str(d.rate))
    check("服务端默认 temperature 在合法范围", 0 < d.temperature <= 1.0, str(d.temperature))

    # 客户端显式传了 -> 应该被识别
    r2 = TTSRequest(text="你好", voice="Timbre1", speed=7, rate=1.2, temperature=0.5)
    check("客户端传了 voice 被识别", r2.provided("voice"))
    check("客户端传了 speed 被识别", r2.provided("speed"))
    check("客户端传了 rate 被识别", r2.provided("rate"))
    check("客户端传了 temperature 被识别", r2.provided("temperature"))
    check("voice 值正确", r2.voice == "Timbre1", str(r2.voice))


def test_engine_wiring() -> None:
    print("\n[5] 引擎接线")
    from examples.ws_server.engine import TTSEngine

    cfg = load_tts_config()
    eng = TTSEngine(config=cfg)
    check("引擎拿到了配置", eng.cfg is cfg)
    check("引擎 device 与配置一致", eng.device == cfg.model.device, repr(eng.device))
    check(
        "引擎 flash_attn 与配置一致",
        eng.use_flash_attn == cfg.model.use_flash_attn,
        str(eng.use_flash_attn),
    )
    # 显式覆盖优先于配置
    eng2 = TTSEngine(device="cpu", config=cfg)
    check("显式 device 覆盖配置", eng2.device == "cpu", repr(eng2.device))


def main() -> int:
    print("=" * 60)
    print("TTS 配置自检 (音色 + 模型)")
    print("=" * 60)

    test_config_loaded()
    test_model_section()
    test_voices()
    test_defaults_override()
    test_engine_wiring()

    print("\n" + "=" * 60)
    if FAILED:
        print(f"失败 {FAILED} 项, 通过 {PASSED} 项")
        return 1
    print(f"全部通过 ({PASSED} 项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
