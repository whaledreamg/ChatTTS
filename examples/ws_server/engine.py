"""ChatTTS 推理引擎封装。

对外只暴露两个能力：

- :meth:`TTSEngine.synthesize` 生成器，流式产出 PCM 片段
- :meth:`TTSEngine.interrupt` 打断当前生成

线程模型：``Chat`` 实例非线程安全，因此本模块设计的实例只在**单个** worker
线程中被访问，服务端通过全局队列保证串行执行。
"""

import os
import sys
import time
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from examples.ws_server.player import AudioPlayer

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

import torch  # noqa: E402

import ChatTTS  # noqa: E402
from examples.ws_server.audio import (  # noqa: E402
    SAMPLE_RATE,
    StreamingPhaseVocoder,
    append_silence,
    ensure_terminal_punctuation,
    estimate_cps,
    is_all_silence,
    smart_fade_out,
    tail_decay_ms,
    trim_leading_silence,
)
from tools.logger import get_logger  # noqa: E402

logger = get_logger("TTSEngine")

#: 预置音色 -> 采样 seed（沿用 examples/web/funcs.py 的表）
VOICE_SEEDS: Dict[str, int] = {
    "Default": 2,
    "Timbre1": 1111,
    "Timbre2": 2222,
    "Timbre3": 3333,
    "Timbre4": 4444,
    "Timbre5": 5555,
    "Timbre6": 6666,
    "Timbre7": 7777,
    "Timbre8": 8888,
    "Timbre9": 9999,
}

#: 流式时每个 batch 生成的 token 数，越小首包延迟越低
STREAM_BATCH = 8

#: 默认语速倍率。1.0 = 使用模型原生语速。
#: ChatTTS 的 [speed_N] token 对语速控制很弱且不单调（实测 speed=0 反而比 speed=2 慢，
#: 整体只有 4.7~7.1 字/秒），所以额外提供一层可控的保音高变速。
DEFAULT_RATE = 1.0

#: 目标语速区间（汉字/秒）。中文自然说话约 4~6 字/秒。
TARGET_CPS_MIN = 4.5
TARGET_CPS_MAX = 6.5

#: 结尾淡出时长（毫秒），避免最后一个字被硬切产生突兀感
FADE_OUT_MS = 40

#: 结尾补静音时长（毫秒），让听感自然收束
TAIL_SILENCE_MS = 120

#: 是否裁掉模型自带的前导静音（ChatTTS 固定带 240~400ms，听感像延迟）
TRIM_LEADING_SILENCE = True

#: 是否自动给没有终止标点的文本补上句号。
#: ChatTTS 在文本无终止标点时会较早产生 EOS，末字发音被硬切；
#: 补标点后末字的韵母能被完整渲染。
#: 可用环境变量 TTS_AUTO_PUNCTUATE=0 关闭。
AUTO_PUNCTUATE = os.environ.get("TTS_AUTO_PUNCTUATE", "1").lower() not in (
    "0",
    "false",
    "no",
    "off",
)

#: 末音节衰减时长低于此值（毫秒）判定为「末字被硬切」。
#: 完整韵母的自然收尾需要 120~250ms，被 EOS 截断通常 < 80ms。
TAIL_DECAY_MIN_MS = 110

#: 测量末音节衰减时的「结束比例」——衰减到峰值的这个比例就算收尾。
#: 默认 2%（而不是早先的 10%）：韵尾（-i/-u/-n 等浊韵尾）是渐弱的，
#: 振幅降到 10% 之后还会再拖 50~150ms，这段是听得见的。
#: 按 10% 判定会把大量发音完整的音频误判成截断，从而触发代价极高的重试。
#: 详见 audio.tail_decay_ms 的说明与 tools/exp_threshold.py 的实测数据。
TAIL_DECAY_END_RATIO = 0.02

#: 末字被硬切时的最大重试次数（每次换随机种子重新采样）
MAX_RETRY = 3

#: 只有文本长度不超过这个值时才启用「末字截断重试」。
#: 重试需要缓冲整段音频才能判定，长文本会牺牲流式低延迟，得不偿失。
RETRY_MAX_CHARS = 20

#: 末字重试的总时间预算（毫秒）。
#:
#: 重试要重新生成整段音频，代价很高；而「末字截断」的判定必须等整段音频
#: 出来才能做，缓冲本身又会把流式退化成阻塞生成。
#:
#: 所以这里加一道时间闸：只有当这一轮已经花掉的时间还在预算内，才值得再试。
#: 超出预算就直接用当前结果（末字略有截断，远好过让用户多等几秒）。
RETRY_TIME_BUDGET_MS = 2500

#: 是否启用「末字截断重试」。
#:
#: 默认**关闭**。实测（tools/exp_retry_value.py）这个功能是净负收益：
#:
#:     平均首帧: 649ms  ->  3415ms  (+2766ms, 4~9 倍)
#:     平均总耗时: 1974ms  ->  3415ms
#:     修好的次数: 1/4     反而弄坏的: 1/4
#:
#: 原因有两点：
#: 1. 判定阈值太敏感，把发音完整的音频也判成截断（已通过放宽
#:    TAIL_DECAY_END_RATIO 修正）。
#: 2. 换随机种子重采样对「末字」只有约 25% 的命中率，且重采样
#:    有时会把原本正常的末字弄坏。
#:
#: 语音场景对延迟极敏感，宁可末字偶尔收得急一点，也不要让用户多等 3 秒。
#: 需要时可置 TTS_RETRY_ON_TRUNCATION=1 打开。
RETRY_ON_TRUNCATION = os.environ.get("TTS_RETRY_ON_TRUNCATION", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def float_to_pcm16(wav: np.ndarray) -> np.ndarray:
    """把 float32 波形转成 PCM16（小端）。

    与 ``tools.audio.np.float_to_int16`` 的区别：这里使用**固定**满量程缩放，
    而不是按块最大值归一化。分块流式输出时，逐块归一化会导致块间音量跳变。
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2")


class TTSEngine:
    """ChatTTS 的线程安全封装（单线程独占）。"""

    def __init__(self, device: Optional[str] = None, use_flash_attn: bool = False):
        self.device = device
        self.use_flash_attn = use_flash_attn
        self.chat: Optional[ChatTTS.Chat] = None
        self.spk_emb_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------ 加载

    def load(self) -> bool:
        """加载模型（本地 asset 目录）与 normalizer。"""
        self.chat = ChatTTS.Chat(get_logger("ChatTTS"))
        ok = self.chat.load(
            source="local",
            device=torch.device(self.device) if self.device else None,
            compile=False,
            use_flash_attn=self.use_flash_attn,
            enable_cache=True,
        )
        if not ok:
            logger.error("failed to load ChatTTS models")
            return False

        try:
            from tools.normalizer.en import normalizer_en_nemo_text

            self.chat.normalizer.register("en", normalizer_en_nemo_text())
        except Exception as exc:  # pragma: no cover - 取决于环境
            logger.warning("register 'en' normalizer failed: %s", exc)

        try:
            from tools.normalizer.zh import normalizer_zh_tn

            self.chat.normalizer.register("zh", normalizer_zh_tn())
        except Exception as exc:  # pragma: no cover - 取决于环境
            logger.warning("register 'zh' normalizer failed: %s", exc)

        self._precompute_voices()
        self._warmup()
        return True

    def _precompute_voices(self) -> None:
        """预先为预置音色生成 spk_emb，避免每个请求重复采样。"""
        assert self.chat is not None
        for name, seed in VOICE_SEEDS.items():
            try:
                torch.manual_seed(seed)
                self.spk_emb_cache[name] = self.chat.sample_random_speaker()
            except Exception as exc:  # pragma: no cover
                logger.warning("sample speaker for %s failed: %s", name, exc)
        logger.info("precomputed %d voices", len(self.spk_emb_cache))

    def _warmup(self) -> None:
        """预热推理链路, 让第一个真实请求不必承担初始化开销。

        不做这一步的话, 第一个请求要额外花 2.6~3.5 秒:
        - NeMo 文本归一化的懒加载 (实测首次 2659ms, 之后 2~3ms)
        - CUDA kernel 的首次 JIT 编译与显存分配

        对语音助手来说, 服务刚起来后的第一句话最容易被用户当成「卡住了」,
        所以这里在 load() 阶段就把这些开销一次性吃掉。
        """
        assert self.chat is not None
        started = time.time()
        try:
            # 一段极短的文本, 只为触发归一化懒加载 + CUDA 预热
            for chunk in self.chat.infer(
                "你好。",
                stream=True,
                lang=None,
                skip_refine_text=True,
                refine_text_only=False,
                use_decoder=True,
                do_text_normalization=True,
                do_homophone_replacement=True,
                split_text=False,
                params_infer_code=ChatTTS.Chat.InferCodeParams(
                    spk_emb=self.resolve_spk_emb("Default", None),
                    prompt="[speed_5]",
                    temperature=0.3,
                    top_P=0.7,
                    top_K=20,
                    stream_batch=STREAM_BATCH,
                    pass_first_n_batches=0,
                    show_tqdm=False,
                ),
                params_refine_text=ChatTTS.Chat.RefineTextParams(show_tqdm=False),
            ):
                _ = chunk
        except Exception as exc:  # pragma: no cover - 预热失败不致命
            logger.warning("warmup failed: %s", exc)
            return
        logger.info("warmup done in %dms", int((time.time() - started) * 1000))

    @property
    def voices(self) -> List[str]:
        return list(VOICE_SEEDS.keys())

    # ------------------------------------------------------------------ 合成

    def resolve_spk_emb(self, voice: Optional[str], spk_emb: Optional[str]) -> Optional[str]:
        """按优先级解析说话人：显式 spk_emb > 预置 voice > 随机。"""
        if spk_emb:
            return spk_emb
        if voice:
            cached = self.spk_emb_cache.get(voice)
            if cached is not None:
                return cached
            logger.warning("unknown voice '%s', fallback to random speaker", voice)
        assert self.chat is not None
        return self.chat.sample_random_speaker()

    def synthesize(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        spk_emb: Optional[str] = None,
        speed: int = 5,
        lang: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 20,
        refine_text: bool = False,
        rate: float = DEFAULT_RATE,
        player: Optional["AudioPlayer"] = None,
        retry_on_truncation: bool = False,
    ) -> Iterator[np.ndarray]:
        """合成语音，流式产出 float32 单声道波形片段。

        参数
        ----
        rate:
            保音高变速倍率，>1 变快、<1 变慢。1.0 表示使用模型原生语速。
        player:
            若传入，则每块音频同时写入声卡，实现边生成边播报。
        retry_on_truncation:
            检测到末字被硬切时是否自动重试。**默认关闭**。

            ChatTTS 的采样有随机性，同一段文本不同次采样的末字完整度
            差异很大（实测末音节衰减在 0~270ms 之间波动）。重试会换
            随机种子重新采样，代价是**首帧延迟翻 4~9 倍**。

            实测这个功能是净负收益（tools/exp_retry_value.py）：
            修好的次数 1/4，反而弄坏的 1/4，平均首帧 +2766ms。
            语音场景对延迟极敏感，所以默认关闭；
            需要时置环境变量 ``TTS_RETRY_ON_TRUNCATION=1``。

        音频后处理（在流式下逐块进行，保证低延迟）：

        1. 前导静音裁剪（流式下需要等待一小段才能判断）
        2. 保音高变速（相位声码器，增量处理）
        3. 结尾淡出 + 补静音（仅在最后一块）
        """
        assert self.chat is not None, "engine not loaded"

        resolved_spk = self.resolve_spk_emb(voice, spk_emb)

        # --- 补终止标点，避免末字被 EOS 截断 ---
        # ChatTTS 在文本没有终止标点时会较早产生 EOS，末字的发声被硬切
        # （实测末音节衰减只有 40ms，而同一字在句中需要 200ms）。
        prepared_text = ensure_terminal_punctuation(text) if AUTO_PUNCTUATE else text

        attempt = 0
        synth_started = time.time()
        while True:
            # 末字是否被硬切？检测到就在结果交付前重试。
            #
            # 注意：重试需要拿到完整音频才能判定，因此这里必须先把整段
            # 缓冲下来。为了不破坏流式低延迟体验，只对**短文本**启用重试
            # （长文本的末字被截断概率低，且重试的代价太大）。
            #
            # 关键：只有需要重试判定时才缓冲整段；否则直接边生成边 yield。
            # 之前无论长短都先 list() 缓冲完，把「流式」退化成了整段阻塞生成，
            # 首帧延迟等于整段合成耗时（实测 9.3 秒）。
            #
            # 再加一道时间闸：重试要重新生成整段音频，代价很高。
            # 只有当这一轮已经花掉的时间还在预算内，才值得再试一次。
            elapsed_ms = (time.time() - synth_started) * 1000
            needs_retry_check = (
                retry_on_truncation
                and AUTO_PUNCTUATE
                and len(prepared_text) <= RETRY_MAX_CHARS
                and elapsed_ms < RETRY_TIME_BUDGET_MS
            )

            if not needs_retry_check:
                # 不需要判定末字截断: 直接流式产出, 保证首帧延迟最低
                for c in self._synthesize_once(
                    prepared_text,
                    resolved_spk=resolved_spk,
                    speed=speed,
                    lang=lang,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    refine_text=refine_text,
                    rate=rate,
                ):
                    if player is not None:
                        player.write_pcm(float_to_pcm16(c).tobytes())
                    yield c
                return

            chunks = list(
                self._synthesize_once(
                    prepared_text,
                    resolved_spk=resolved_spk,
                    speed=speed,
                    lang=lang,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    refine_text=refine_text,
                    rate=rate,
                )
            )
            if not chunks:
                return

            decay = tail_decay_ms(
                np.concatenate(chunks), end_ratio=TAIL_DECAY_END_RATIO
            )
            if decay < TAIL_DECAY_MIN_MS:
                # 再试一次之前先看时间预算: 超了就用当前结果,
                # 末字略有截断远好过让用户多等几秒。
                spent_ms = (time.time() - synth_started) * 1000
                if attempt < MAX_RETRY and spent_ms < RETRY_TIME_BUDGET_MS:
                    attempt += 1
                    logger.info(
                        "tail truncated (decay=%.0fms < %dms), retry #%d "
                        "(spent=%.0fms)",
                        decay,
                        TAIL_DECAY_MIN_MS,
                        attempt,
                        spent_ms,
                    )
                    # 换随机种子重新采样
                    torch.manual_seed(
                        int(torch.randint(0, 2**31 - 1, (1,)).item())
                    )
                    continue
                logger.warning(
                    "tail still truncated (decay=%.0fms, spent=%.0fms), "
                    "不再重试",
                    decay,
                    spent_ms,
                )

            for c in chunks:
                if player is not None:
                    player.write_pcm(float_to_pcm16(c).tobytes())
                yield c
            return

    def _synthesize_once(
        self,
        text: str,
        *,
        resolved_spk: Optional[str],
        speed: int,
        lang: Optional[str],
        temperature: float,
        top_p: float,
        top_k: int,
        refine_text: bool,
        rate: float,
    ) -> Iterator[np.ndarray]:
        """执行一次合成，产出后处理好的音频块（不负责播放）。"""
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=resolved_spk,
            prompt=f"[speed_{speed}]",
            temperature=temperature,
            top_P=top_p,
            top_K=top_k,
            stream_batch=STREAM_BATCH,
            pass_first_n_batches=0,
            show_tqdm=False,
        )

        params_refine_text = ChatTTS.Chat.RefineTextParams(
            show_tqdm=False,
        )

        generator = self.chat.infer(
            text,
            stream=True,
            lang=lang,
            skip_refine_text=not refine_text,
            refine_text_only=False,
            use_decoder=True,
            do_text_normalization=True,
            do_homophone_replacement=True,
            split_text=False,
            params_infer_code=params_infer_code,
            params_refine_text=params_refine_text,
        )

        vocoder = StreamingPhaseVocoder(factor=rate) if rate else None
        pending = np.zeros(0, dtype=np.float32)
        # 等待前导静音判定所需的最少样本数
        lookahead = int(0.6 * SAMPLE_RATE) if TRIM_LEADING_SILENCE else 0
        trimmed = not TRIM_LEADING_SILENCE
        # 保留一块缓冲，用于给最后一块做淡出
        held: Optional[np.ndarray] = None
        # 是否已经收到「生成结束」标记
        exhausted = False

        for chunk in generator:
            if chunk is None:
                continue
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if not arr.size:
                # ChatTTS 在流式结束时会给一个空块，作为结束标记
                exhausted = True
                continue

            # --- 1. 前导静音裁剪 ---
            if not trimmed:
                pending = np.concatenate([pending, arr])
                # 不用死等 lookahead 到满: 一旦确认「已经进入发声段」就立刻裁剪。
                #
                # ChatTTS 的前导静音固定在 240~400ms, lookahead 设 600ms 是为了
                # 留余量。但死等满 600ms 会让首帧延迟凭空多出几百毫秒
                # (实测 GPU 出第一个块只要 125ms, 首帧却有 650ms, 差额就在这)。
                #
                # 这里改成「自适应」: 每收到一块就检查一次是否已经越过静音段,
                # 一旦越过就立即裁剪并输出, 不再等满 lookahead。
                # 只有「到目前为止全是静音」时才继续等。
                if pending.size < lookahead and is_all_silence(pending):
                    continue
                arr = trim_leading_silence(pending)
                trimmed = True
                pending = np.zeros(0, dtype=np.float32)

            # --- 2. 保音高变速 ---
            if vocoder is not None:
                arr = vocoder.process(arr)

            if not arr.size:
                continue

            # 缓冲一块，保证能在最后一块上做淡出
            if held is not None:
                yield held
            held = arr

        # --- 收尾：冲刷声码器 ---
        if vocoder is not None:
            flushed = vocoder.flush()
            if flushed.size:
                if held is not None:
                    held = np.concatenate([held, flushed])
                else:
                    held = flushed

        # --- 前导静音裁剪的兜底 ---
        # 整段音频比 lookahead 还短时（例如「你好」只有 ~330ms），
        # 循环结束时 pending 里还存着全部音频但从未被裁剪输出，
        # 会导致语音整段丢失。这里必须把 pending 补上。
        if not trimmed and pending.size:
            arr = trim_leading_silence(pending)
            pending = np.zeros(0, dtype=np.float32)
            trimmed = True
            if vocoder is not None:
                arr = vocoder.process(arr)
            if arr.size:
                if held is not None:
                    held = np.concatenate([held, arr])
                else:
                    held = arr

        # --- 最后一块做淡出，让末字自然收束 ---
        if held is not None:
            held = smart_fade_out(held, FADE_OUT_MS)
            yield held

        # 补一段尾部静音，让听感自然结束
        silence = append_silence(np.zeros(0, dtype=np.float32), TAIL_SILENCE_MS)
        if silence.size:
            yield silence

    def interrupt(self) -> None:
        """打断当前正在进行的生成。"""
        if self.chat is not None:
            self.chat.interrupt()
