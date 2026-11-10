"""TTS 服务端配置 (configs/tts.yaml)。

**为什么要有这一层**: 音色和模型之前是硬编码在 `engine.py` 里的
(`VOICE_SEEDS` 字典 + `load(source="local")`), 改音色必须改代码。
现在都从 yaml 读, 改音色只改配置。

配置文件里有两个概念, 别混淆:

- **模型** (`model`) —— 加载哪套权重、跑在什么设备上。服务端启动时生效。
- **音色** (`voices`) —— 有哪些可选音色。客户端在 WS 报文里用
  `"voice": "名字"` 选。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - 依赖缺失时给出清晰提示
    yaml = None

logger = logging.getLogger("tts.config")

#: 项目根目录 (Bot Agent/) —— 从 services/tts_service/examples/ws_server/ 上溯
ROOT = Path(__file__).resolve().parents[4]
CONFIG_DIR = ROOT / "configs"
TTS_CONFIG_PATH = CONFIG_DIR / "tts.yaml"


@dataclass
class ModelConfig:
    """模型加载参数。"""

    #: local / huggingface / custom
    source: str = "local"
    #: source=custom 时的模型目录
    custom_path: str = ""
    #: 推理设备 ("" = 自动)
    device: str = ""
    compile: bool = False
    use_flash_attn: bool = False
    enable_cache: bool = True
    use_vllm: bool = False
    experimental: bool = False
    coef: str = ""


@dataclass
class VoiceConfig:
    """一个音色的定义。"""

    #: 客户端用的名字
    name: str = ""
    #: 随机种子 (与 ref_audio 二选一, 或互为兜底)
    seed: Optional[int] = None
    #: 参考音频路径 (相对项目根目录或绝对路径) —— 用于克隆音色
    ref_audio: str = ""
    #: 参考音频文本 (留空 = 自动转写)
    ref_text: str = ""


@dataclass
class SynthDefaults:
    """合成默认参数 (客户端没给时用这些)。"""

    voice: str = "Default"
    speed: int = 5
    rate: float = 1.0
    lang: Optional[str] = None
    temperature: float = 0.3
    top_p: float = 0.7
    top_k: int = 20
    refine_text: bool = False


@dataclass
class TTSServiceConfig:
    """TTS 服务端配置总表。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    voices: List[VoiceConfig] = field(default_factory=list)
    defaults: SynthDefaults = field(default_factory=SynthDefaults)
    #: 是否成功读到过配置文件
    loaded: bool = False

    @property
    def voice_names(self) -> List[str]:
        return [v.name for v in self.voices if v.name]

    def get_voice(self, name: str) -> Optional[VoiceConfig]:
        for v in self.voices:
            if v.name == name:
                return v
        return None


def _resolve_path(raw: str) -> str:
    """把配置里的相对路径解析成绝对路径。

    相对路径按**项目根目录**算 —— 这样配置里写 `assets/voices/a.wav`
    不受「从哪个目录启动」影响。
    """
    if not raw:
        return ""
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    return str((ROOT / path).resolve())


def load_tts_config(path: Path = TTS_CONFIG_PATH) -> TTSServiceConfig:
    """加载 TTS 服务端配置。

    文件缺失时返回**内置默认值** (等价于原来的 `VOICE_SEEDS` + `load(local)`),
    所以即使没有 `configs/tts.yaml`, 服务也能正常起来。
    """
    data: Dict[str, Any] = {}
    if yaml is not None and path and Path(path).is_file():
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - 配置坏了不该让服务起不来
            logger.warning("[TTS] 读取 %s 失败, 用内置默认: %s", path, exc)
            data = {}
    elif yaml is None:
        logger.warning("[TTS] 缺少 pyyaml 依赖, 用内置默认配置")

    # ---- 模型 ----
    m = data.get("model") or {}
    model = ModelConfig(
        source=str(m.get("source", "local") or "local"),
        custom_path=_resolve_path(str(m.get("custom_path", "") or "")),
        device=str(m.get("device", "") or os.environ.get("TTS_DEVICE", "")),
        compile=bool(m.get("compile", False)),
        use_flash_attn=bool(m.get("use_flash_attn", False)),
        enable_cache=bool(m.get("enable_cache", True)),
        use_vllm=bool(m.get("use_vllm", False)),
        experimental=bool(m.get("experimental", False)),
        coef=str(m.get("coef", "") or ""),
    )

    # ---- 音色 ----
    voices: List[VoiceConfig] = []
    for item in data.get("voices") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "") or "").strip()
        if not name:
            logger.warning("[TTS] 有条音色定义缺 name, 跳过")
            continue
        seed = item.get("seed")
        voices.append(
            VoiceConfig(
                name=name,
                seed=None if seed is None else int(seed),
                ref_audio=_resolve_path(str(item.get("ref_audio", "") or "")),
                ref_text=str(item.get("ref_text", "") or ""),
            )
        )

    # ---- 合成默认值 ----
    d = data.get("defaults") or {}
    lang_raw = str(d.get("lang", "") or "").strip().lower()
    defaults = SynthDefaults(
        voice=str(d.get("voice", "Default") or "Default"),
        speed=int(d.get("speed", 5)),
        rate=float(d.get("rate", 1.0)),
        lang=lang_raw or None,
        temperature=float(d.get("temperature", 0.3)),
        top_p=float(d.get("top_p", 0.7)),
        top_k=int(d.get("top_k", 20)),
        refine_text=bool(d.get("refine_text", False)),
    )

    cfg = TTSServiceConfig(
        model=model, voices=voices, defaults=defaults, loaded=bool(data)
    )

    if not voices:
        logger.warning("[TTS] 配置里没有音色, 回落到内置默认音色")
    logger.info(
        "[TTS] 配置已加载: model=%s device=%s | 音色 %d 个: %s | 默认音色 %s",
        model.source,
        model.device or "auto",
        len(voices),
        ", ".join(cfg.voice_names) or "(无)",
        defaults.voice,
    )
    return cfg
