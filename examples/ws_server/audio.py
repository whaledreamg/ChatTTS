"""音频后处理工具。

解决两个常见问题：

1. **语速不自然**：ChatTTS 的 ``[speed_N]`` token 对语速控制很弱且不单调
   （实测 speed=0 反而比 speed=2 慢，整体只有 4.7~7.1 字/秒），模型还会额外
   产生 240~400ms 的前导静音。本模块提供：

   - :func:`trim_leading_silence` 裁掉前导静音
   - :class:`StreamingPhaseVocoder` 保音高变速（相位声码器，流式安全）

2. **结尾突兀**：最后一个字被硬切，听感发毛。本模块提供：

   - :func:`fade_out` 结尾淡出
   - :func:`append_silence` 结尾补静音
"""

from typing import List, Optional

import numpy as np

#: ChatTTS 采样率
SAMPLE_RATE = 24000

#: 判定「静音」的幅度阈值
SILENCE_THRESH = 1e-3


# ---------------------------------------------------------------- 基础工具


def is_all_silence(
    wav: np.ndarray,
    threshold: float = SILENCE_THRESH,
) -> bool:
    """判断一段音频是否**全部**是静音。

    用于流式裁剪前导静音时的「提前放行」判断：如果当前收到的这段
    音频里已经出现了发声，就可以立刻裁剪输出，不必继续等 lookahead
    攒满。

    注意阈值比 :func:`voiced_ratio` 用得宽松一些 —— 这里只需要判断
    「有没有开始发声」，不需要精确的比例。
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return True
    return not bool(np.any(np.abs(arr) > threshold))


def trim_leading_silence(
    wav: np.ndarray,
    threshold: float = SILENCE_THRESH,
    preroll_ms: int = 10,
) -> np.ndarray:
    """裁掉开头多余的静音（ChatTTS 固定带 240~400ms 前导静音，听感上像延迟）。"""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    nz = np.nonzero(np.abs(arr) > threshold)[0]
    if len(nz) == 0:
        return arr
    # 保留一点点缓冲，避免切掉音头
    start = max(0, int(nz[0]) - int(preroll_ms * SAMPLE_RATE / 1000))
    return arr[start:]


def fade_out(wav: np.ndarray, ms: int = 40) -> np.ndarray:
    """结尾做一段线性淡出，避免最后一个字被硬切。"""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = min(int(ms * SAMPLE_RATE / 1000), arr.size)
    if n <= 0:
        return arr
    out = arr.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def smart_fade_out(
    wav: np.ndarray,
    ms: int = 25,
    silence_ratio: float = 0.02,
) -> np.ndarray:
    """只在音频确实是「硬切」时才淡出，否则原样返回。

    ChatTTS 输出的末尾本身通常已经自然衰减到接近静音（末样本幅度普遍
    < 峰值的 1%）。此时再做固定时长的淡出，会把末字仍处于发声状态的
    尾巴也一起削掉，听感就是「最后一个字没说完」。

    这里改成：先判断末尾是否已经自然收束（末样本幅度低于峰值的
    ``silence_ratio``），是则完全不动；否则只在最后那点残留能量上做
    淡出，并把更早的样本清零，避免削到有效发声。
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return arr
    peak = float(np.abs(arr).max())
    if peak < 1e-9:
        return arr

    # 已经自然收束：末样本幅度足够小，不需要淡出
    if float(np.abs(arr[-1])) <= silence_ratio * peak:
        return arr

    # 找到最后一个仍然「响」的样本，只对它之后的残留做淡出
    thr = silence_ratio * peak
    idx = np.nonzero(np.abs(arr) > thr)[0]
    if len(idx) == 0:
        return arr
    last = int(idx[-1])
    n = min(int(ms * SAMPLE_RATE / 1000), last + 1)
    if n <= 0:
        return arr
    out = arr.copy()
    out[last - n + 1 : last + 1] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    out[last + 1 :] = 0.0
    return out


def append_silence(wav: np.ndarray, ms: int = 120) -> np.ndarray:
    """结尾补一段静音，让听感自然收束。"""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = int(ms * SAMPLE_RATE / 1000)
    if n <= 0:
        return arr
    return np.concatenate([arr, np.zeros(n, dtype=np.float32)])


def voiced_ratio(wav: np.ndarray, threshold: float = SILENCE_THRESH) -> float:
    """有声样本占比。"""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float((np.abs(arr) > threshold).mean())


#: 视为「句子已自然收尾」的标点
TERMINAL_PUNCTUATION = "。！？!?"

#: 视为「句子中间停顿」的标点
CLAUSE_PUNCTUATION = "，、；：,;:～~"


def ensure_terminal_punctuation(text: str) -> str:
    """确保文本以终止标点结尾。

    ChatTTS 的 GPT 在文本没有终止标点时会较早产生 EOS token，导致
    **最后一个字的发声被截断**（实测「今天天气不错」的末音节从峰值
    衰减到 10% 只需要 40ms，而同一个字放在句中需要 200ms）。

    补上终止标点后，模型会把末字的韵母完整渲染出来。

    注意：已经带终止标点或以省略号结尾的文本不做改动。
    """
    s = (text or "").strip()
    if not s:
        return text
    # 已经以终止标点结尾
    if s[-1] in TERMINAL_PUNCTUATION:
        return s
    # 省略号结尾（如「等等…」），本身已有收束语义
    if s.endswith("...") or s.endswith("…"):
        return s
    # 结尾是语气词/助词时，用逗号比句号更自然，但同样能让模型继续发声
    if s[-1] in CLAUSE_PUNCTUATION:
        return s
    return s + "。"


def tail_decay_ms(
    wav: np.ndarray,
    win_ms: int = 10,
    end_ratio: float = 0.02,
) -> float:
    """测量末音节的自然衰减时长（毫秒）。

    做法：在 RMS 包络上找到最后一个局部峰，然后计算从该峰衰减到
    「峰自身幅度的 ``end_ratio``」所需的时间。

    - 末字被 EOS 截断：通常 < 80ms（听起来就是「字没说完就断了」）
    - 末字完整（韵母自然收尾）：通常 120~250ms

    ``end_ratio`` 的默认值是 **2%** 而不是 10%。

    为什么改：韵母（尤其是 -i/-u/-n 这类浊韵尾）的自然收尾是渐弱的，
    振幅从峰值降到 10% 之后还会再拖 50~150ms 才归零。这段尾巴**是听得见的**
    （决定了「嗯」收得干不干净），但按 10% 判定会把大量发音完整的音频
    误判成「末字被硬切」。

    实测（见 tools/exp_threshold.py）同一批音频在不同 end_ratio 下的判定：

        文本                              10%      5%      3%      2%
        你好，今天天气不错。              75      75      80      80
        北京今天天气晴，气温 24 度。        55      55      55      60
        请告诉我你在哪个城市。             30     225     225     395
        建议您打开手机自带的天气应用查看。    45      60     220     225

    「请告诉我你在哪个城市。」这类韵尾饱满的句子，在 10% 下只有 30ms
    （判为截断），在 2% 下有 395ms（判为正常）—— 同一段音频，结论完全相反。

    注意：尾部补的静音不影响本指标——静音段本来就低于阈值，
    峰的位置和衰减计算都发生在有声段内。
    """
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return 0.0
    w = max(1, int(win_ms * SAMPLE_RATE / 1000))
    n = arr.size // w
    if n < 3:
        return 0.0
    env = np.sqrt((arr[: n * w].reshape(n, w) ** 2).mean(axis=1))

    # 找最后一个局部峰
    peaks: List[int] = []
    i = 1
    while i < len(env) - 1:
        if env[i] >= env[i - 1] and env[i] > env[i + 1] and env[i] > 0.10:
            peaks.append(i)
            i += 2
        else:
            i += 1
    # 合并过近的峰
    merged: List[int] = []
    for p in peaks:
        if merged and (p - merged[-1]) * win_ms < 80:
            if env[p] > env[merged[-1]]:
                merged[-1] = p
        else:
            merged.append(p)
    if not merged:
        return 0.0

    c = merged[-1]
    peak = float(env[c])
    if peak <= 1e-9:
        return 0.0
    thr = end_ratio * peak
    for d in range(0, 600):
        j = c + d
        if j >= len(env):
            return d * win_ms
        if env[j] < thr:
            return d * win_ms
    return 600


def estimate_cps(wav: np.ndarray, n_chars: int) -> float:
    """估算有效语速（字/秒），只统计有声部分。"""
    arr = np.asarray(wav, dtype=np.float32).reshape(-1)
    if arr.size == 0 or n_chars <= 0:
        return 0.0
    nz = np.nonzero(np.abs(arr) > SILENCE_THRESH)[0]
    if len(nz) == 0:
        return 0.0
    voiced = (int(nz[-1]) - int(nz[0])) / SAMPLE_RATE
    if voiced <= 0:
        return 0.0
    return n_chars / voiced


# ------------------------------------------------------- 保音高变速（相位声码器）


class StreamingPhaseVocoder:
    """流式保音高变速器（相位声码器）。

    ``factor > 1`` 变快，``factor < 1`` 变慢。音高保持不变。

    采用标准 overlap-add 相位声码器：分析 hop = ``hop_out * factor``，
    合成 hop = ``hop_out``，每帧相位按「输入相位偏差」累加，
    因此可以在收到音频块时增量处理，无需整段缓冲。
    """

    def __init__(
        self,
        factor: float = 1.0,
        n_fft: int = 1024,
        hop_out: int = 256,
    ):
        self.factor = float(factor)
        self.n_fft = n_fft
        self.hop_out = hop_out
        self.hop_in = max(1, int(round(hop_out * self.factor)))
        self._ratio = self.hop_out / self.hop_in
        self.window = np.hanning(n_fft).astype(np.float32)
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_pos = 0
        self._ola = np.zeros(n_fft, dtype=np.float32)
        self._ola_w = np.zeros(n_fft, dtype=np.float32)
        self._prev_phase: Optional[np.ndarray] = None
        self._out_phase: Optional[np.ndarray] = None
        self._omega = 2.0 * np.pi * np.arange(n_fft // 2 + 1) / n_fft
        self._frames = 0

    def _process_frame(self, frame: np.ndarray) -> None:
        X = np.fft.rfft(frame * self.window)
        mag = np.abs(X)
        ph = np.angle(X)

        if self._prev_phase is None:
            out_ph = ph.copy()
        else:
            # 输入帧相隔 hop_in，期望相位推进 omega * hop_in
            expected = self._prev_phase + self._omega * self.hop_in
            dphase = ph - expected
            dphase -= 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
            # dphase 携带了「真实频率 - 频点中心频率」的偏差。
            # 输出帧相隔 hop_out，要保持音高不变，输出相位必须按真实频率推进：
            #     2*pi*f*hop_out/SR = omega*hop_out + dphase*(hop_out/hop_in)
            # 少了 (hop_out/hop_in) 这个系数会导致频率偏移、波形相消、幅度衰减。
            out_ph = self._out_phase + self._omega * self.hop_out + dphase * self._ratio

        self._prev_phase = ph
        self._out_phase = out_ph

        y = np.fft.irfft(mag * np.exp(1j * out_ph), n=self.n_fft).astype(np.float32)
        self._ola += y * self.window
        self._ola_w += self.window * self.window
        self._frames += 1

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """输入一块音频，返回变速后的音频块。"""
        if abs(self.factor - 1.0) < 1e-3:
            return np.asarray(chunk, dtype=np.float32).reshape(-1)

        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size:
            self._in_buf = np.concatenate([self._in_buf, data])

        outs: List[np.ndarray] = []
        while len(self._in_buf) - self._in_pos >= self.n_fft:
            frame = self._in_buf[self._in_pos : self._in_pos + self.n_fft]
            self._process_frame(frame)
            self._in_pos += self.hop_in
            # 当前帧的 hop_out 个样本已完整，可以输出
            denom = self._ola_w[: self.hop_out]
            out = self._ola[: self.hop_out] / np.maximum(denom, 1e-8)
            outs.append(out.astype(np.float32))
            # 左移 hop_out
            self._ola[: self.n_fft - self.hop_out] = self._ola[self.hop_out :]
            self._ola[self.n_fft - self.hop_out :] = 0.0
            self._ola_w[: self.n_fft - self.hop_out] = self._ola_w[self.hop_out :]
            self._ola_w[self.n_fft - self.hop_out :] = 0.0

        if self._in_pos > 0:
            self._in_buf = self._in_buf[self._in_pos :]
            self._in_pos = 0

        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """把缓冲区里剩余的样本处理完（补零到 n_fft）。"""
        if abs(self.factor - 1.0) < 1e-3:
            return np.zeros(0, dtype=np.float32)
        remaining = self._in_buf.size
        if remaining == 0:
            return np.zeros(0, dtype=np.float32)
        # 补零让最后一帧能被处理
        padded = np.concatenate(
            [self._in_buf, np.zeros(self.n_fft, dtype=np.float32)]
        )
        self._in_buf = padded
        outs: List[np.ndarray] = []
        while len(self._in_buf) - self._in_pos >= self.n_fft:
            frame = self._in_buf[self._in_pos : self._in_pos + self.n_fft]
            self._process_frame(frame)
            self._in_pos += self.hop_in
            denom = self._ola_w[: self.hop_out]
            out = self._ola[: self.hop_out] / np.maximum(denom, 1e-8)
            outs.append(out.astype(np.float32))
            self._ola[: self.n_fft - self.hop_out] = self._ola[self.hop_out :]
            self._ola[self.n_fft - self.hop_out :] = 0.0
            self._ola_w[: self.n_fft - self.hop_out] = self._ola_w[self.hop_out :]
            self._ola_w[self.n_fft - self.hop_out :] = 0.0
        self._in_buf = self._in_buf[self._in_pos :]
        self._in_pos = 0
        out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
        # 只保留与原始等长的部分（按比例）
        expected = int(remaining / self.factor)
        if out.size > expected:
            out = out[:expected]
        return out


def change_speed(wav: np.ndarray, factor: float) -> np.ndarray:
    """整段保音高变速（非流式）。"""
    if abs(factor - 1.0) < 1e-3:
        return np.asarray(wav, dtype=np.float32).reshape(-1)
    voc = StreamingPhaseVocoder(factor=factor)
    out = voc.process(np.asarray(wav, dtype=np.float32).reshape(-1))
    tail = voc.flush()
    if tail.size:
        out = np.concatenate([out, tail])
    return out.astype(np.float32)
