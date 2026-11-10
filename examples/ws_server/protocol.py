"""WebSocket TTS 协议定义。

文本帧（JSON）用于控制，音频通过二进制帧推送。

二进制帧结构（小端定长头 + 可变负载）::

    magic(4s)    = b"TTS1"
    type(1B)     帧类型
    flags(1B)    附加标记（音频编码格式等）
    seq(4u)      帧序号，从 0 开始按请求递增
    sample_rate(4u)
    n_samples(4u)  音频采样点数（每声道）
    req_id_len(2u)
    req_id(req_id_len bytes, utf-8)
    payload      剩余字节

帧类型::

    1 = AUDIO_DATA   音频数据
    2 = AUDIO_END    该 request_id 的音频流结束

flags 位（仅 AUDIO_DATA 有意义）::

    0x01 = MP3
    0x02 = WAV
    0x00 = PCM16LE（默认）
"""

import json
import struct
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

MAGIC = b"TTS1"

FRAME_TYPE_AUDIO_DATA = 1
FRAME_TYPE_AUDIO_END = 2

FORMAT_PCM = "pcm"
FORMAT_MP3 = "mp3"
FORMAT_WAV = "wav"

SUPPORTED_FORMATS = (FORMAT_PCM, FORMAT_MP3, FORMAT_WAV)

FLAG_MP3 = 0x01
FLAG_WAV = 0x02

#: 头部长度：magic(4) + type(1) + flags(1) + seq(4) + sample_rate(4) + n_samples(4) + req_id_len(2)
HEADER_STRUCT = struct.Struct("<4sBBIIHH")
HEADER_SIZE = HEADER_STRUCT.size

#: 单帧最大请求 id 长度（字节）
MAX_REQ_ID_LEN = 255


class ProtocolError(ValueError):
    """协议层错误，携带面向客户端的错误信息。"""

    def __init__(self, message: str, code: str = "bad_request") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def format_flags(fmt: str) -> int:
    if fmt == FORMAT_MP3:
        return FLAG_MP3
    if fmt == FORMAT_WAV:
        return FLAG_WAV
    return 0


def flags_format(flags: int) -> str:
    if flags & FLAG_MP3:
        return FORMAT_MP3
    if flags & FLAG_WAV:
        return FORMAT_WAV
    return FORMAT_PCM


class TTSRequest(BaseModel):
    """客户端发来的合成请求。

    **客户端没显式传的字段走服务端默认** (``configs/tts.yaml`` 的
    ``defaults`` 段)。用 ``model_fields_set`` 判断哪些字段是客户端真的
    传了的 —— 否则客户端不传 voice 时会被 pydantic 的字段默认值
    ("Default") 盖掉服务端的配置。
    """

    type: str = "tts"
    request_id: Optional[str] = None
    text: str
    voice: Optional[str] = None
    spk_emb: Optional[str] = None
    speed: int = 5
    lang: Optional[str] = None
    format: str = FORMAT_PCM
    stream: bool = True
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    refine_text: bool = False
    play: bool = False
    rate: float = 1.0
    #: 是否启用「末字截断重试」。None = 用服务端默认值（默认关闭）。
    #: 打开会让首帧延迟翻 4~9 倍，命中率只有 ~25%。
    retry_on_truncation: Optional[bool] = None

    def provided(self, field: str) -> bool:
        """客户端是否显式传了这个字段。"""
        return field in self.model_fields_set

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("text must not be empty")
        return v

    @field_validator("format")
    @classmethod
    def _check_format(cls, v: str) -> str:
        v = (v or FORMAT_PCM).lower()
        if v not in SUPPORTED_FORMATS:
            raise ValueError(
                f"unsupported format: {v}, expected one of {SUPPORTED_FORMATS}"
            )
        return v

    @field_validator("speed")
    @classmethod
    def _check_speed(cls, v: int) -> int:
        if v < 0 or v > 9:
            raise ValueError("speed must be in [0, 9]")
        return v

    @field_validator("lang")
    @classmethod
    def _check_lang(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.lower()
        if v not in ("zh", "en"):
            raise ValueError("lang must be 'zh' or 'en'")
        return v

    @field_validator("temperature")
    @classmethod
    def _check_temperature(cls, v: float) -> float:
        if v <= 0 or v > 1.0:
            raise ValueError("temperature must be in (0, 1]")
        return v

    @field_validator("top_p")
    @classmethod
    def _check_top_p(cls, v: float) -> float:
        if v <= 0 or v > 1.0:
            raise ValueError("top_p must be in (0, 1]")
        return v

    @field_validator("top_k")
    @classmethod
    def _check_top_k(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("top_k must be in [1, 100]")
        return v

    @field_validator("rate")
    @classmethod
    def _check_rate(cls, v: float) -> float:
        if v < 0.5 or v > 2.0:
            raise ValueError("rate must be in [0.5, 2.0]")
        return v


class ControlMessage(BaseModel):
    """非合成类的控制消息（stop / ping）。"""

    type: str
    request_id: Optional[str] = None

    @field_validator("type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        if v not in ("stop", "ping", "cancel"):
            raise ValueError(f"unsupported control type: {v}")
        return v


def parse_json_message(raw: str) -> Dict[str, Any]:
    """解析客户端 JSON 文本帧。"""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json: {exc}", code="invalid_json") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("json message must be an object", code="invalid_json")
    return obj


def parse_control(raw: str) -> ControlMessage:
    obj = parse_json_message(raw)
    try:
        return ControlMessage(**obj)
    except Exception as exc:
        raise ProtocolError(str(exc), code="bad_request") from exc


def parse_request(raw: str) -> TTSRequest:
    obj = parse_json_message(raw)
    obj.setdefault("type", "tts")
    if obj.get("type") != "tts":
        raise ProtocolError(
            f"unsupported message type: {obj.get('type')}", code="bad_request"
        )
    try:
        return TTSRequest(**obj)
    except Exception as exc:
        raise ProtocolError(str(exc), code="bad_request") from exc


def pack_audio_frame(
    payload: bytes,
    *,
    seq: int,
    request_id: str,
    sample_rate: int,
    n_samples: int,
    fmt: str = FORMAT_PCM,
) -> bytes:
    """打包一个音频二进制帧。"""
    req_id = (request_id or "").encode("utf-8")
    if len(req_id) > MAX_REQ_ID_LEN:
        req_id = req_id[:MAX_REQ_ID_LEN]
    header = HEADER_STRUCT.pack(
        MAGIC,
        FRAME_TYPE_AUDIO_DATA,
        format_flags(fmt),
        seq,
        sample_rate,
        n_samples,
        len(req_id),
    )
    return header + req_id + payload


def pack_end_frame(
    *,
    seq: int,
    request_id: str,
    sample_rate: int = 0,
) -> bytes:
    """打包一个音频结束帧。"""
    req_id = (request_id or "").encode("utf-8")
    if len(req_id) > MAX_REQ_ID_LEN:
        req_id = req_id[:MAX_REQ_ID_LEN]
    header = HEADER_STRUCT.pack(
        MAGIC,
        FRAME_TYPE_AUDIO_END,
        0,
        seq,
        sample_rate,
        0,
        len(req_id),
    )
    return header + req_id


def unpack_frame(data: bytes) -> Tuple[int, int, int, int, str, bytes]:
    """解析二进制帧，返回 (type, flags, seq, sample_rate, request_id, payload)。"""
    if len(data) < HEADER_SIZE:
        raise ProtocolError("frame too short", code="invalid_frame")
    magic, ftype, flags, seq, sample_rate, n_samples, req_id_len = HEADER_STRUCT.unpack(
        data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ProtocolError("bad magic", code="invalid_frame")
    body = data[HEADER_SIZE:]
    if len(body) < req_id_len:
        raise ProtocolError("truncated frame", code="invalid_frame")
    request_id = body[:req_id_len].decode("utf-8", errors="replace")
    payload = body[req_id_len:]
    if ftype == FRAME_TYPE_AUDIO_DATA and fmt_is_pcm(flags):
        # PCM 每个采样点 2 字节
        if len(payload) % 2 != 0:
            raise ProtocolError("odd pcm payload length", code="invalid_frame")
    return ftype, flags, seq, sample_rate, request_id, payload


def fmt_is_pcm(flags: int) -> bool:
    return flags & (FLAG_MP3 | FLAG_WAV) == 0
