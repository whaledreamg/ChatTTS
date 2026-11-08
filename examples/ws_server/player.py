"""服务端本地音频播报。

把 ChatTTS 生成的音频直接通过声卡播放出来，用于「服务端收到报文就出声」的场景。

实现方式：

- PCM：启动 ``aplay`` 子进程，把 PCM 字节持续写入其 stdin，实现边生成边播报
- MP3 / WAV：先写成临时文件，再用 ``ffplay`` 播放

如果系统里没有 ``aplay`` / ``ffplay``，会降级为静默（只记日志），不影响服务本身。
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional

from tools.logger import get_logger

logger = get_logger("AudioPlayer")


class AudioPlayer:
    """在服务端把音频播放到声卡。"""

    def __init__(self, sample_rate: int = 24000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._proc: Optional[subprocess.Popen] = None
        self._aplay = shutil.which("aplay")
        self._ffplay = shutil.which("ffplay")
        self._ffmpeg = shutil.which("ffmpeg")
        if not self._aplay and not self._ffplay:
            logger.warning(
                "no aplay/ffplay found, server-side playback is disabled"
            )

    # ------------------------------------------------------------------ PCM

    @property
    def available(self) -> bool:
        """是否有可用的音频输出方式。"""
        return bool(self._aplay or self._ffplay)

    def open_pcm_stream(self) -> bool:
        """打开 PCM 输出流，准备写入。"""
        self.close()
        if self._aplay:
            cmd = [
                self._aplay,
                "-q",
                "-r",
                str(self.sample_rate),
                "-f",
                "S16_LE",
                "-c",
                str(self.channels),
                "-t",
                "raw",
                "-",
            ]
        elif self._ffplay:
            cmd = [
                self._ffplay,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-f",
                "s16le",
                "-ar",
                str(self.sample_rate),
                "-ac",
                str(self.channels),
                "-i",
                "pipe:0",
            ]
        else:
            return False

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            logger.error("failed to start audio output: %s", exc)
            self._proc = None
            return False
        return True

    def write_pcm(self, data: bytes) -> None:
        """写入 PCM 字节（阻塞写，天然起到流控作用）。"""
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            logger.warning("audio output pipe closed unexpectedly")
            self.close()

    def close(self) -> None:
        """关闭输出流。"""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ----------------------------------------------------------------- 文件

    def play_file(self, path: str) -> bool:
        """播放一个音频文件（mp3 / wav）。"""
        if self._ffplay:
            cmd = [
                self._ffplay,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                path,
            ]
        elif self._aplay:
            cmd = [self._aplay, "-q", path]
        else:
            return False
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            logger.error("failed to play %s: %s", path, exc)
            return False
        return True

    def play_bytes(self, blob: bytes, suffix: str) -> bool:
        """把编码后的音频写临时文件后播放。"""
        if not blob:
            return False
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(blob)
            return self.play_file(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
