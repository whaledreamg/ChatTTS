"""ChatTTS WebSocket 服务。

启动::

    conda run -n chattts python examples/ws_server/main.py
    # 或
    conda run -n chattts uvicorn examples.ws_server.main:app --host 0.0.0.0 --port 8000

协议见同目录 README.md。
"""

import asyncio
import contextlib
import os
import sys
import threading
import time
from typing import Any, Dict, Optional

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

import numpy as np  # noqa: E402
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from examples.ws_server.engine import (  # noqa: E402
    RETRY_ON_TRUNCATION,
    SAMPLE_RATE,
    TTSEngine,
    float_to_pcm16,
)
from examples.ws_server.player import AudioPlayer  # noqa: E402
from examples.ws_server.protocol import (  # noqa: E402
    FORMAT_MP3,
    FORMAT_PCM,
    FORMAT_WAV,
    ProtocolError,
    pack_audio_frame,
    pack_end_frame,
    parse_control,
    parse_request,
)
from tools.logger import get_logger  # noqa: E402

logger = get_logger("WS-Server")

engine = TTSEngine()

#: 服务端是否默认本地播报（可用环境变量 TTS_PLAY=1 开启）
#: 报文里带 "play": true 时无论此设置如何都会播报
DEFAULT_PLAY = os.environ.get("TTS_PLAY", "0").lower() in ("1", "true", "yes", "on")

#: 全局待处理任务队列（单 worker 串行推理）
PENDING: "asyncio.Queue[Any]" = asyncio.Queue()

#: 当前正在执行的任务，用于 stop/cancel
CURRENT_TASK: Dict[str, Any] = {}

#: 服务端本地播报器（单 worker 独占，天然线程安全）
player = AudioPlayer(sample_rate=SAMPLE_RATE)


def _encode_wav(pcm: np.ndarray) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def _encode_mp3(pcm: np.ndarray) -> bytes:
    from tools.audio import pcm_arr_to_mp3_view

    return bytes(pcm_arr_to_mp3_view(pcm.astype(np.float32) / 32767.0))


class Task:
    """一次合成请求在服务端的表示。"""

    def __init__(self, request, out_queue: "asyncio.Queue[Any]", request_id: str):
        self.request = request
        self.out_queue = out_queue
        self.request_id = request_id
        self.cancelled = False
        # 是否在服务端本地播报
        self.play = bool(getattr(request, "play", False)) or DEFAULT_PLAY
        # 进入队列的时刻, 用来算排队等待时长
        self.queued_at = time.time()
        self.queued_ms = 0


def _push_text(task: Task, payload: Dict[str, Any]) -> None:
    task.out_queue.put_nowait(("text", payload))


def _push_bytes(task: Task, payload: bytes) -> None:
    task.out_queue.put_nowait(("bytes", payload))


def _synthesize_pcm(task: Task):
    """按请求参数合成，产出 PCM16 片段。"""
    req = task.request
    for wav in engine.synthesize(
        req.text,
        voice=req.voice,
        spk_emb=req.spk_emb,
        speed=req.speed,
        lang=req.lang,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        refine_text=req.refine_text,
        rate=req.rate,
        player=player if task.play else None,
        # 默认关闭末字截断重试: 它会让首帧延迟翻 4~9 倍, 而命中率只有 ~25%。
        # 客户端可以按需显式打开。
        retry_on_truncation=bool(
            req.retry_on_truncation if req.retry_on_truncation is not None
            else RETRY_ON_TRUNCATION
        ),
    ):
        if task.cancelled:
            return
        pcm = float_to_pcm16(wav)
        if pcm.size:
            yield pcm


def _run_task_sync(task: Task, loop: asyncio.AbstractEventLoop) -> None:
    """在工作线程中执行推理，把结果投递回事件循环。

    该函数是阻塞的，同一时刻只会有一个（全局单 worker）。
    """
    req = task.request
    seq = 0
    started = time.time()

    def emit_audio(pcm: np.ndarray) -> None:
        nonlocal seq
        payload = pcm.astype("<i2").tobytes()
        frame = pack_audio_frame(
            payload,
            seq=seq,
            request_id=task.request_id,
            sample_rate=SAMPLE_RATE,
            n_samples=int(pcm.size),
            fmt=req.format,
        )
        seq += 1
        loop.call_soon_threadsafe(_push_bytes, task, frame)

    # 首个音频帧的时刻, 用来算"生成首帧"耗时
    first_frame_at = None
    try:
        if req.format == FORMAT_PCM and req.stream:
            # 流式：边生成边推
            if task.play:
                player.open_pcm_stream()
            for pcm in _synthesize_pcm(task):
                if first_frame_at is None:
                    first_frame_at = time.time()
                emit_audio(pcm)
            if task.play:
                player.close()
        else:
            # 整段模式：合成完毕后一次性打包
            chunks = list(_synthesize_pcm(task))
            if chunks:
                merged = np.concatenate(chunks)
                first_frame_at = time.time()
                if req.format == FORMAT_MP3:
                    blob = _encode_mp3(merged)
                elif req.format == FORMAT_WAV:
                    blob = _encode_wav(merged)
                else:  # pragma: no cover - 已在协议层校验
                    blob = merged.astype("<i2").tobytes()
                frame = pack_audio_frame(
                    blob,
                    seq=0,
                    request_id=task.request_id,
                    sample_rate=SAMPLE_RATE,
                    n_samples=int(merged.size),
                    fmt=req.format,
                )
                loop.call_soon_threadsafe(_push_bytes, task, frame)
                if task.play:
                    suffix = ".mp3" if req.format == FORMAT_MP3 else ".wav"
                    player.play_bytes(blob, suffix)
    except Exception as exc:  # pragma: no cover - 兜底
        logger.exception("inference failed")
        if task.play:
            player.close()
        loop.call_soon_threadsafe(
            _push_text,
            task,
            {"type": "error", "request_id": task.request_id, "message": str(exc)},
        )

    end_frame = pack_end_frame(
        seq=seq,
        request_id=task.request_id,
        sample_rate=SAMPLE_RATE,
    )
    loop.call_soon_threadsafe(_push_bytes, task, end_frame)

    total_ms = int((time.time() - started) * 1000)
    first_ms = (
        int((first_frame_at - started) * 1000) if first_frame_at is not None else -1
    )
    logger.info(
        "[TTS-服务端] 总计 %dms | 首帧 %dms | 队列等待 %dms | chars=%d",
        total_ms,
        first_ms,
        task.queued_ms,
        len(req.text),
    )
    loop.call_soon_threadsafe(
        _push_text,
        task,
        {
            "type": "done",
            "request_id": task.request_id,
            "duration_ms": total_ms,
            "first_frame_ms": first_ms,
            "queued_ms": task.queued_ms,
        },
    )


async def worker_loop() -> None:
    """全局唯一的推理 worker，保证 ChatTTS 串行访问。"""
    while True:
        task: Task = await PENDING.get()
        if task.cancelled:
            PENDING.task_done()
            continue
        CURRENT_TASK["current"] = task
        # 从入队到被 worker 取出的等待时长
        task.queued_ms = int((time.time() - task.queued_at) * 1000)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _run_task_sync, task, loop)
        except Exception as exc:  # pragma: no cover
            logger.exception("worker task crashed")
            with contextlib.suppress(Exception):
                task.out_queue.put_nowait(
                    (
                        "text",
                        {
                            "type": "error",
                            "request_id": task.request_id,
                            "message": str(exc),
                        },
                    )
                )
        finally:
            CURRENT_TASK.pop("current", None)
            PENDING.task_done()


async def sender_loop(websocket: WebSocket, out_queue: "asyncio.Queue[Any]") -> None:
    """把出站队列里的消息写到 socket 上（唯一写者，避免并发写冲突）。"""
    while True:
        kind, payload = await out_queue.get()
        try:
            if kind == "text":
                await websocket.send_json(payload)
            else:
                await websocket.send_bytes(payload)
        except (WebSocketDisconnect, RuntimeError) as exc:
            # 客户端已断开，发送协程直接退出
            logger.info("websocket sender stopped: %s", exc)
            return
        except Exception as exc:
            logger.info("websocket send failed: %s", exc)
            return


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading ChatTTS models ...")
    ok = await asyncio.get_running_loop().run_in_executor(None, engine.load)
    if not ok:
        logger.error("model load failed, server will not serve requests")
    else:
        logger.info("ChatTTS ready, voices: %s", engine.voices)
    worker = asyncio.create_task(worker_loop())
    try:
        yield
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker


app = FastAPI(title="ChatTTS WebSocket Server", lifespan=lifespan)


@app.get("/health")
async def health():
    return JSONResponse(
        {
            "status": "ok" if engine.chat is not None else "loading",
            "model_loaded": engine.chat is not None,
            "queue_size": PENDING.qsize(),
            "sample_rate": SAMPLE_RATE,
            "voices": engine.voices,
            "server_playback": player.available,
            "default_play": DEFAULT_PLAY,
        }
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    out_queue: "asyncio.Queue[Any]" = asyncio.Queue()
    sender = asyncio.create_task(sender_loop(websocket, out_queue))
    tasks: Dict[str, Task] = {}

    out_queue.put_nowait(
        (
            "text",
            {
                "type": "ready",
                "sample_rate": SAMPLE_RATE,
                "channels": 1,
                "formats": [FORMAT_PCM, FORMAT_MP3, FORMAT_WAV],
                "default_format": FORMAT_PCM,
                "voices": engine.voices,
                "server_playback": player.available,
                "default_play": DEFAULT_PLAY,
            },
        )
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            text_msg = message.get("text")
            binary_msg = message.get("bytes")

            if text_msg is None and binary_msg is None:
                continue

            if binary_msg is not None:
                out_queue.put_nowait(
                    (
                        "text",
                        {"type": "error", "message": "binary input is not supported"},
                    )
                )
                continue

            try:
                control = parse_control(text_msg)
            except ProtocolError as exc:
                # 可能是 tts 请求，parse_control 会拒绝未知 type
                try:
                    request = parse_request(text_msg)
                except ProtocolError as req_exc:
                    out_queue.put_nowait(
                        (
                            "text",
                            {
                                "type": "error",
                                "code": req_exc.code,
                                "message": req_exc.message,
                            },
                        )
                    )
                    continue
                request_id = request.request_id or f"r{len(tasks)}"
                if request_id in tasks:
                    out_queue.put_nowait(
                        (
                            "text",
                            {
                                "type": "error",
                                "code": "duplicate_request_id",
                                "request_id": request_id,
                                "message": f"request_id '{request_id}' already in flight",
                            },
                        )
                    )
                    continue
                task = Task(request, out_queue, request_id)
                tasks[request_id] = task
                await PENDING.put(task)
                out_queue.put_nowait(
                    (
                        "text",
                        {
                            "type": "accepted",
                            "request_id": request_id,
                            "queue_pos": PENDING.qsize(),
                            "format": request.format,
                            "stream": request.format == FORMAT_PCM and request.stream,
                        },
                    )
                )
                continue

            if control.type == "ping":
                out_queue.put_nowait(("text", {"type": "pong"}))
            elif control.type in ("stop", "cancel"):
                target_id = control.request_id
                if target_id is None:
                    current = CURRENT_TASK.get("current")
                    target_id = current.request_id if current else None
                if target_id and target_id in tasks:
                    tasks[target_id].cancelled = True
                engine.interrupt()
                out_queue.put_nowait(
                    (
                        "text",
                        {
                            "type": "interrupted",
                            "request_id": target_id,
                        },
                    )
                )
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover
        logger.exception("websocket loop error: %s", exc)
    finally:
        for task in tasks.values():
            task.cancelled = True
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        # 客户端可能已断开，发送结束帧会抛异常，这里统一吞掉
        with contextlib.suppress(Exception):
            await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("TTS_HOST", "0.0.0.0"),
        port=int(os.environ.get("TTS_PORT", "8000")),
        log_level="info",
    )
