#!/usr/bin/env python3
"""生成 ASR replay 测试用 wav (16kHz 单声道)。

两种来源:
1. TTS 合成 (默认): 用 ChatTTS 合成 "唤醒词" + 间隔 + "问句"
2. 真实录音: --real <问句录音.wav> 把真人录音接在唤醒词后面

用法:
    conda run -n chattts python tools/make_test_wav.py --out /tmp/test_16k.wav
    conda run -n chattts python tools/make_test_wav.py --out /tmp/test_16k.wav \
        --real /home/gxh/workspace/project/ASRT/recordings/rec_20261107_102622.wav
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MAGIC = b"TTS1"
HEADER_STRUCT = struct.Struct("<4sBBIIHH")
FRAME_TYPE_AUDIO_DATA = 1
FRAME_TYPE_AUDIO_END = 2


def unpack_frame(data: bytes):
    magic, ftype, flags, seq, rate, n_samples, rid_len = HEADER_STRUCT.unpack(
        data[:HEADER_STRUCT.size]
    )
    if magic != MAGIC:
        raise ValueError("bad magic")
    body = data[HEADER_STRUCT.size:]
    return ftype, flags, seq, rate, body[:rid_len], body[rid_len:]


def _synthesize(ws_url: str, text: str, voice: str):
    from websockets.sync.client import connect

    chunks = []
    sample_rate = 24000
    with connect(ws_url, max_size=None, open_timeout=10) as ws:
        ws.send(
            json.dumps(
                {
                    "type": "tts",
                    "request_id": "make-test-wav",
                    "text": text,
                    "voice": voice,
                    "speed": 5,
                    "format": "pcm",
                    "stream": True,
                    "play": False,
                },
                ensure_ascii=False,
            )
        )
        while True:
            message = ws.recv(timeout=180)
            if isinstance(message, str):
                continue
            ftype, flags, seq, sr, rid, payload = unpack_frame(message)
            if ftype == FRAME_TYPE_AUDIO_END:
                break
            if ftype == FRAME_TYPE_AUDIO_DATA and payload:
                sample_rate = sr
                chunks.append(payload)
    if not chunks:
        raise RuntimeError("没有合成出音频")
    return b"".join(chunks), sample_rate


def read_wav_pcm(path: Path):
    """读 wav 的 (pcm, 采样率)。"""
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(f"{path} 必须是 16bit 单声道")
        return wf.readframes(wf.getnframes()), wf.getframerate()


def write_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def resample(src: Path, dst: Path, rate: int = 16000) -> bool:
    """优先用 ffmpeg, 回退到纯 python 实现。"""
    if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0:
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-acodec", "pcm_s16le", "-ar", str(rate), "-ac", "1", str(dst),
        ]
        return subprocess.run(cmd, capture_output=True).returncode == 0

    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        print(f"缺少重采样依赖: {exc}")
        return False

    data, orig_sr = sf.read(str(src), always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1, keepdims=True)
    if orig_sr != rate:
        from math import gcd

        g = gcd(rate, orig_sr)
        data = resample_poly(data, rate // g, orig_sr // g, axis=0)
    sf.write(str(dst), data, rate, subtype="PCM_16")
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/asr_test_16k.wav")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    p.add_argument("--voice", default="Default")
    p.add_argument("--wake", default="你好助手")
    p.add_argument("--question", default="今天天气怎么样")
    p.add_argument("--real", default=None, help="用真实录音作为问句部分")
    p.add_argument("--skip-wake-audio", action="store_true",
                   help="不合成唤醒词 (配合 ASR --no-wake 使用)")
    p.add_argument("--lead", type=float, default=0.4, help="前置静音秒数")
    p.add_argument("--gap", type=float, default=1.5, help="唤醒词与问句之间的间隔秒数")
    args = p.parse_args()

    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # 唤醒词 (TTS 合成)
        if args.skip_wake_audio:
            print("跳过唤醒词合成")
            wake_pcm, rate = b"", 24000
        else:
            print(f"合成唤醒词: [{args.wake}]")
            wake_pcm, rate = _synthesize(args.url, args.wake, args.voice)
        wake_wav = tmpdir / "wake.wav"
        write_wav(wake_wav, wake_pcm, rate)

        # 问句部分
        if args.real:
            print(f"问句使用真实录音: {args.real}")
            real_pcm, real_rate = read_wav_pcm(Path(args.real))
            q_wav = tmpdir / "q.wav"
            write_wav(q_wav, real_pcm, real_rate)
            q_rate = real_rate
        else:
            print(f"合成问句: [{args.question}]")
            q_pcm, q_rate = _synthesize(args.url, args.question, args.voice)
            q_wav = tmpdir / "q.wav"
            write_wav(q_wav, q_pcm, q_rate)

        # 两段各自重采样到 16k, 再在 16k 上拼接
        wake16 = tmpdir / "wake16.wav"
        if not resample(wake_wav, wake16, 16000):
            print("唤醒词重采样失败")
            return 1
        q16 = tmpdir / "q16.wav"
        if not resample(q_wav, q16, 16000):
            print("问句重采样失败")
            return 1

        # 拼成: 前置静音 + 唤醒词 + 间隔静音 + 问句
        lead = b"\x00\x00" * int(16000 * args.lead)
        gap = b"\x00\x00" * int(16000 * args.gap)
        wake_data, _ = read_wav_pcm(wake16)
        q_data, _ = read_wav_pcm(q16)
        write_wav(out, lead + wake_data + gap + q_data, 16000)

    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
