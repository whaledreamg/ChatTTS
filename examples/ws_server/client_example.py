"""ChatTTS WebSocket 服务测试客户端。

三种用法：

1. 拼装请求（常用）::

       conda run -n chattts python examples/ws_server/client_example.py \
           --text "你好，世界" --voice Timbre1 --out output.wav

2. **直接发送原始报文**（最贴近真实客户端，方便在终端里手写 JSON）::

       conda run -n chattts python examples/ws_server/client_example.py \
           --json '{"type":"tts","text":"你好","voice":"Timbre1"}'

   报文里的字段会被原样发给服务端，客户端不再做校验，方便构造异常报文做测试。

3. 发送控制报文::

       conda run -n chattts python examples/ws_server/client_example.py \
           --json '{"type":"ping"}' --no-out
"""

import argparse
import asyncio
import io
import json
import os
import sys
import wave

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

import websockets  # noqa: E402

from examples.ws_server.protocol import unpack_frame  # noqa: E402


def pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


async def run(
    uri: str,
    payload: str,
    out_path: str,
    save: bool,
    quiet: bool,
    recv_timeout: float = 120.0,
) -> None:
    """连接服务端，发送一条报文，打印收到的所有帧。"""
    pcm = bytearray()
    blob = bytearray()
    frames = 0

    async with websockets.connect(uri, max_size=None) as ws:
        await ws.send(payload)
        if not quiet:
            print(f">> {payload}")

        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
            except asyncio.TimeoutError:
                if not quiet:
                    print("!! recv timeout, aborting")
                break
            if isinstance(message, bytes):
                ftype, flags, seq, sample_rate, request_id, data = unpack_frame(message)
                if ftype == 1:
                    frames += 1
                    if not quiet:
                        print(
                            f"<< [binary] audio seq={seq} bytes={len(data)} "
                            f"rate={sample_rate} req={request_id}"
                        )
                    if data:
                        blob.extend(data)
                        pcm.extend(data)
                elif ftype == 2:
                    if not quiet:
                        print(f"<< [binary] end of audio req={request_id}")
                continue

            if not quiet:
                print(f"<< {message}")
            try:
                obj = json.loads(message)
            except (TypeError, ValueError):
                obj = {"type": "unknown"}
            kind = obj.get("type")
            if kind in ("done", "error", "pong"):
                # 终态帧：结束本次交互
                break

    if save and out_path:
        if pcm:
            with open(out_path, "wb") as f:
                f.write(pcm_to_wav(bytes(pcm)))
            if not quiet:
                print(f"saved {out_path} ({frames} audio frames)")
        else:
            with open(out_path, "wb") as f:
                f.write(bytes(blob))
            if not quiet:
                print(f"saved {out_path}")
    return


def build_payload(args: argparse.Namespace) -> str:
    """根据命令行参数拼装请求 JSON。"""
    if args.json:
        # 原样透传，不做校验，方便构造异常报文
        return args.json
    request = {
        "type": "tts",
        "text": args.text,
        "voice": args.voice,
        "speed": args.speed,
        "format": args.format,
    }
    if args.lang:
        request["lang"] = args.lang
    if args.request_id:
        request["request_id"] = args.request_id
    if args.play:
        request["play"] = True
    if args.rate != 1.0:
        request["rate"] = args.rate
    return json.dumps(request, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChatTTS WebSocket 服务测试客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--uri", default="ws://127.0.0.1:8000/ws", help="服务端地址")
    parser.add_argument(
        "--json",
        default=None,
        help="直接发送原始 JSON 报文（优先于 --text 等参数）",
    )
    parser.add_argument("--text", default="你好，世界，这是一段测试语音。", help="合成文本")
    parser.add_argument("--voice", default="Default", help="音色名")
    parser.add_argument("--request-id", default=None, help="请求 id")
    parser.add_argument("--speed", type=int, default=5, help="语速 0-9")
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="保音高变速倍率，>1 变快、<1 变慢（0.5~2.0）",
    )
    parser.add_argument("--lang", default=None, help="语言 zh/en")
    parser.add_argument(
        "--format", default="pcm", choices=["pcm", "mp3", "wav"], help="音频格式"
    )
    parser.add_argument("--out", default="ws_output.wav", help="音频输出文件")
    parser.add_argument(
        "--no-out", action="store_true", help="不保存音频文件（只打印帧信息）"
    )
    parser.add_argument("--quiet", action="store_true", help="不打印收到的帧")
    parser.add_argument(
        "--play", action="store_true", help="请求服务端在本地扬声器播报"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="接收超时（秒），默认 120"
    )
    args = parser.parse_args()

    payload = build_payload(args)
    asyncio.run(
        run(
            args.uri,
            payload,
            args.out,
            not args.no_out,
            args.quiet,
            args.timeout,
        )
    )


if __name__ == "__main__":
    main()
