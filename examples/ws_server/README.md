# ChatTTS WebSocket 服务

把仓库内的 ChatTTS 封装成一个 WebSocket 服务：客户端发 JSON 描述要合成的文本，
服务端串行排队推理，音频以二进制帧流式推回。

## 启动

```bash
# 在仓库根目录执行
conda run -n chattts python examples/ws_server/main.py

# 或用 uvicorn
conda run -n chattts uvicorn examples.ws_server.main:app --host 0.0.0.0 --port 8000
```

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TTS_HOST` | `0.0.0.0` | 监听地址 |
| `TTS_PORT` | `8000` | 监听端口 |
| `TTS_PLAY` | `0` | 设为 `1` 时，**所有请求都在服务端本地扬声器播报** |

模型权重默认从仓库根目录的 `asset/` 加载（`source="local"`），无需联网。

## 配置（音色 + 模型）

**音色和模型都从 `configs/tts.yaml` 读** —— 换音色 / 换模型不用改代码，
改配置重启服务即可。

```yaml
model:
  source: local          # local / huggingface / custom
  custom_path: ""        # source=custom 时的模型目录
  device: ""             # "" = 自动（有 CUDA 用 CUDA）
  compile: false
  use_flash_attn: false
  enable_cache: true

voices:
  - name: Default        # 客户端写 "voice": "Default"
    seed: 2              # 方式 1: 随机种子（同机同硬件一致）
  - name: 客服小美
    ref_audio: assets/voices/xiaomei.wav   # 方式 2: 参考音频克隆（跨机器一致）

defaults:
  voice: Default         # 默认音色
  speed: 5
  rate: 1.0              # 保音高变速倍率，>1 更快
  temperature: 0.3
  top_p: 0.7
  top_k: 20
```

### 两种音色定义方式

| 方式 | 字段 | 优点 | 缺点 |
|---|---|---|---|
| 随机种子 | `seed: 2` | 轻量、快、加载无额外开销 | **换机器 / CPU↔GPU 回退时音色会变** |
| 参考音频克隆 | `ref_audio: xxx.wav` | **跨机器一致** | 启动时多一次 DVAE 编码，慢一点 |

> 种子音色在 CPU 与 GPU 上结果不同 —— ChatTTS 的采样器 CPU/GPU 用不同的
> 随机数流。这就是「之前重启后音色变了」的原因。要音色稳定就用
> `ref_audio`。

两个都写时：**优先克隆，失败回落到 seed**。

### 客户端与服务端的优先级

**客户端没显式传的字段走服务端默认值**（`configs/tts.yaml` 的 `defaults` 段）。

| 字段 | 服务端默认 | 客户端覆盖 |
|---|---|---|
| `voice` | `defaults.voice` | 报文里带 `"voice": "Timbre3"` |
| `speed` / `rate` | `defaults.speed` / `defaults.rate` | 报文里带 `"speed": 7` |
| `temperature` / `top_p` / `top_k` | `defaults.*` | 报文里带 `"temperature": 0.5` |

客户端传了不存在的音色 → **回落到服务端默认音色**（并记一条 WARN 日志），
不会失败。

### 查当前生效的配置

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

```json
{"status":"ok","model_loaded":true,"queue_size":0,"sample_rate":24000,
 "voices":["Default","Timbre1", ...],
 "server_playback":true,"default_play":false,
 "model":{"source":"local","device":"auto","compile":false},
 "default_voice":"Default",
 "defaults":{"speed":5,"rate":1.0,"temperature":0.3,"top_p":0.7,"top_k":20,"lang":null}}
```

- `server_playback`：系统里是否可用 `aplay` / `ffplay`（决定能否本地播报）
- `default_play`：当前是否默认对所有请求播报
- `default_voice`：当前默认音色
- `model`：模型来源与设备

## 自检

```bash
cd services/tts_service
conda run -n chattts python examples/ws_server/check_tts_config.py    # 配置读取 (36 项)
conda run -n chattts python examples/ws_server/check_tts_behavior.py  # 覆盖逻辑 (23 项)
```

两个脚本都**不需要连服务、不需要加载模型**，纯离线检查。

## 测试客户端

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --text "你好，世界" --voice Timbre1 --out output.wav
```

也可以直接发原始报文（详见下方「终端快速测试」）：

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"你好","voice":"Timbre1"}'
```

## 终端快速测试

以下命令均在**仓库根目录**执行，需先启动服务。

### 1. 健康检查

```bash
curl -s http://127.0.0.1:8000/health
```

### 2. 终端发报文（推荐）

用 `client_example.py --json` 直接把一条 JSON 报文发出去，最接近真实客户端的行为：

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"你好，世界","voice":"Timbre1"}'
```

输出：

```
>> {"type":"tts","text":"你好，世界","voice":"Timbre1"}
<< {"type":"ready","sample_rate":24000,"channels":1,...}
<< {"type":"accepted","request_id":"r0","queue_pos":1,"format":"pcm","stream":true}
<< [binary] audio seq=0 bytes=7680 rate=24000 req=r0
<< [binary] audio seq=1 bytes=8192 rate=24000 req=r0
<< [binary] end of audio req=r0
<< {"type":"done","request_id":"r0","duration_ms":533}
```

`--json` 的内容会**原样透传**给服务端，客户端不做任何校验，因此可以构造异常报文做测试：

```bash
# ping
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"ping"}' --no-out

# 非法 JSON
conda run -n chattts python examples/ws_server/client_example.py \
    --json 'not json' --no-out

# 空文本
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":""}' --no-out

# 非法格式
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"hi","format":"ogg"}' --no-out
```

常用选项：

| 选项 | 说明 |
| --- | --- |
| `--json '<报文>'` | 直接发送原始 JSON 报文（优先于其他参数） |
| `--text` / `--voice` / `--speed` / `--lang` / `--format` | 拼装请求时的字段 |
| `--rate` | 保音高变速倍率，>1 变快、<1 变慢 |
| `--out <文件>` | 音频落盘路径（PCM 会自动包成 WAV） |
| `--no-out` | 不落盘，只打印帧信息 |
| `--quiet` | 不打印收到的帧 |
| `--timeout <秒>` | 接收超时，默认 120 |

### 3. 保存音频到文件

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"你好","voice":"Timbre1"}' \
    --out output.wav
```

### 4. 让服务端本地播报

服务端可以一边生成一边通过声卡播放（客户端仍然会收到音频流）。

**方式一：服务启动时全局开启**（所有请求都播报）

```bash
TTS_PLAY=1 conda run -n chattts python examples/ws_server/main.py
```

**方式二：单条报文指定**（`play` 字段，优先级高于全局设置）

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"你好","play":true}' --no-out
```

或者用命令行开关：

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --text "你好" --play --no-out
```

> 本地播报依赖系统的 `aplay`（ALSA）或 `ffplay`。若两者都没有，服务会正常返回音频，
> 只是不发出声音（`/health` 的 `server_playback` 会显示 `false`）。

### 5. 交互式命令行客户端

`websockets` 自带 CLI，可直接连上去手敲报文（Ctrl-D 退出）：

```bash
conda run --no-capture-output -n chattts python -m websockets ws://127.0.0.1:8000/ws
```

连上后会看到 `ready` 帧，然后逐行输入 JSON：

```
{"type":"tts","text":"你好，世界","voice":"Timbre1"}
{"type":"ping"}
{"type":"stop"}
```

二进制音频帧会显示成 `(binary) 54545331...` 这样的十六进制串（`54545331` 即帧头魔数 `TTS1`）。

> 注意：`conda run` 默认会捕获输出，交互式程序需要加 `--no-capture-output`。
> 用管道时还要让 stdin 保持打开，否则连接会提前关闭、看不到响应：

```bash
{ echo '{"type":"ping"}'; sleep 3; } \
  | conda run --no-capture-output -n chattts python -m websockets ws://127.0.0.1:8000/ws
```

### 6. 用 `websocat`（可选）

如果环境里装了 `websocat`：

```bash
websocat -t ws://127.0.0.1:8000/ws
```

## 协议

### 文本帧（JSON）

客户端 → 服务端：

```json
{
  "type": "tts",
  "request_id": "abc",
  "text": "你好，世界",
  "voice": "Timbre1",
  "spk_emb": null,
  "speed": 5,
  "lang": "zh",
  "format": "pcm",
  "stream": true,
  "temperature": 0.3,
  "top_p": 0.7,
  "top_k": 20,
  "refine_text": false,
  "play": false,
  "rate": 1.0
}
```

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `type` | str | `"tts"` | 固定 `"tts"` |
| `request_id` | str | 自动生成 | 请求标识，出现在二进制帧头里 |
| `text` | str | 必填 | 要合成的文本，不能为空 |
| `voice` | str | `null` | 预置音色名：`Default`、`Timbre1`..`Timbre9` |
| `spk_emb` | str | `null` | 直接指定 ChatTTS speaker embedding（优先级高于 `voice`） |
| `speed` | int | `5` | 语速 0-9，对应 `[speed_N]` prompt（模型原生控制，力度较弱且不单调） |
| `rate` | float | `1.0` | **保音高变速**倍率，0.5~2.0。>1 变快、<1 变慢，这是实际生效的语速控制 |
| `lang` | str | `null` | `zh` / `en`，`null` 表示自动检测 |
| `format` | str | `"pcm"` | `pcm` / `mp3` / `wav` |
| `stream` | bool | `true` | `pcm` 时是否流式推送；`false` 则整段返回 |
| `temperature` | float | `0.3` | 采样温度，(0, 1] |
| `top_p` | float | `0.7` | nucleus 采样阈值 |
| `top_k` | int | `20` | top-k 采样 |
| `refine_text` | bool | `false` | 是否先做文本润色（增加延迟）。注意：实测润色（单字化）对末字完整性**没有帮助**，默认关闭 |
| `play` | bool | `false` | 是否在**服务端本地扬声器**播报（`TTS_PLAY=1` 时默认为 true） |

> **关于末字完整性**：服务端会自动给没有终止标点的文本补上句号
> （`AUTO_PUNCTUATE`），这是保证「最后一个字完整说完」的关键。
> 如果你想要原始行为，可以设置环境变量 `TTS_AUTO_PUNCTUATE=0`。

其他控制消息：

```json
{"type": "ping"}
{"type": "stop", "request_id": "abc"}
{"type": "cancel", "request_id": "abc"}
```

- `ping` → 回 `{"type":"pong"}`
- `stop` / `cancel`：打断当前生成（`request_id` 省略时打断正在执行的那一个）

服务端 → 客户端：

```json
{"type":"ready","sample_rate":24000,"channels":1,
 "formats":["pcm","mp3","wav"],"default_format":"pcm","voices":[...]}
{"type":"accepted","request_id":"abc","queue_pos":0,"format":"pcm","stream":true}
{"type":"done","request_id":"abc","duration_ms":1234}
{"type":"error","code":"bad_request","message":"...","request_id":"abc"}
{"type":"interrupted","request_id":"abc"}
{"type":"pong"}
```

### 二进制帧

小端定长头 + 变长负载：

| 偏移 | 字段 | 大小 | 说明 |
| --- | --- | --- | --- |
| 0 | `magic` | 4 | 固定 `TTS1` |
| 4 | `type` | 1 | `1` = 音频数据，`2` = 音频结束 |
| 5 | `flags` | 1 | 音频编码：`0` = PCM16LE，`0x01` = MP3，`0x02` = WAV |
| 6 | `seq` | 4 | 帧序号，从 0 递增 |
| 10 | `sample_rate` | 4 | 采样率，恒为 24000 |
| 14 | `n_samples` | 4 | 音频采样点数（每声道） |
| 18 | `req_id_len` | 2 | 请求 id 字节长度 |
| 20 | `req_id` | 可变 | UTF-8 请求 id |
| - | `payload` | 可变 | 音频数据 |

- `type=1`：`payload` 是编码后的音频字节。PCM 时为 16bit 小端有符号整数，单声道。
- `type=2`：`payload` 为空，表示该 `request_id` 的音频流结束。

`format=mp3` / `format=wav` 时只推送一个 `type=1` 帧（整段音频），随后跟 `type=2` 结束帧。

### 交互时序

```
client                                   server
  |---- WS connect ----------------------->|
  |<--- {"type":"ready"} ------------------|
  |---- {"type":"tts","text":"..."} ------>|
  |<--- {"type":"accepted"} ---------------|
  |<--- binary audio frame (seq=0) --------|
  |<--- binary audio frame (seq=1) --------|
  |    ...                                |
  |<--- binary end frame ------------------|
  |<--- {"type":"done"} -------------------|
```

## 语速与结尾优化

服务端在模型输出之后会做一层音频后处理（`examples/ws_server/audio.py`），
解决两个常见问题。

### 语速

ChatTTS 的 `[speed_N]` token 对语速的控制**很弱且不单调**——实测：

| `speed` | 0 | 2 | 5 | 7 | 9 |
| --- | --- | --- | --- | --- | --- |
| 有效语速（字/秒） | 4.68 | 5.97 | 5.99 | 6.66 | 7.13 |

`speed=0` 反而比 `speed=2` 慢。而且模型固定带 **240~400ms 的前导静音**，
听感上像「反应慢了一拍」。

因此服务端提供 `rate` 参数，在模型输出之后做一次**保音高变速**（相位声码器）：

| `rate` | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 |
| --- | --- | --- | --- | --- | --- | --- |
| 有效语速（字/秒） | 4.70 | 5.27 | 5.89 | 6.46 | 7.03 | 7.62 |

中文自然说话约 4~6 字/秒，所以默认 `rate=1.0` 就在自然区间。
想要更快/更慢就调 `rate`：

```bash
conda run -n chattts python examples/ws_server/client_example.py \
    --json '{"type":"tts","text":"今天天气不错","rate":1.2}'
```

前置静音同时会被裁掉（前导静音 263ms → 10ms）。

> `rate` 使用 STFT 相位声码器，**音高保持不变**（实测 440Hz 输入变速后仍为 440.3Hz），
> 且是流式增量处理，不会增加首包延迟。

### 结尾突兀

「最后一个字没完整说完就结束了」这个问题实际上有**两个独立原因**，服务端都做了处理。

#### 原因一：模型在无终止标点时提前 EOS（主因）

ChatTTS 的 GPT 在文本没有终止标点时会较早产生 EOS token，导致**末字的发声被硬切**。

用同一个字做对照实验（测量末音节从峰值衰减到自身 10% 所需时间）：

| 末字「错」的位置 | 衰减时长 |
| --- | --- |
| 句尾（`今天天气不错`） | **40ms**（被削） |
| 句中（`今天天气不错啊`） | **200ms**（完整） |

完整韵母的自然收尾需要 120~250ms，只有 40ms 明显是被切断的。

服务端的做法是**自动给没有终止标点的文本补上句号**（`ensure_terminal_punctuation`），
让模型把末字完整渲染出来。实测末音节衰减时长：

| | 平均 | 中位 | 达标率(≥110ms) |
| --- | --- | --- | --- |
| 修复前 | 77.1ms | 65.0ms | 5/24 |
| 修复后 | 96.7ms | 95.0ms | 7/24 |

> 官方 `refine_text`（把文本打散成单字）实测反而更差（平均 69.6ms，达标 3/24），
> 所以没有采用。

#### 原因三：模型采样有随机性

ChatTTS 的采样是随机的，同一段文本不同次采样，末字的完整度差异很大
（实测末音节衰减在 **0~270ms** 之间波动）。仅靠补标点无法保证每次都完整。

因此服务端增加了**自动重试**：合成完成后检测末音节衰减时长，若低于 110ms
判定为被硬切，就换随机种子重新采样，最多重试 3 次。

由于判定需要完整音频，重试只对**短文本**（≤20 字）启用，以免牺牲长文本的
流式低延迟。

实测（9 个文本 × 8 次采样）：

| | 平均 | 中位 | 达标率(≥110ms) |
| --- | --- | --- | --- |
| 无重试 | 101.1ms | 100.0ms | 32/72 |
| **有重试** | **162.2ms** | **180.0ms** | **64/72** |

#### 原因二：固定时长的淡出削掉了仍在发声的尾巴

ChatTTS 输出的末尾**本身通常已经自然衰减到接近静音**（末样本幅度普遍 < 峰值的 1%）。
此时再对最后 40ms 做固定淡出，会把末字仍处于发声状态的尾巴一起削掉。

原来的实现：

```
修复前(固定淡出): 110:0.0141 100:0.0422 90:0.1654 80:0.1291 ...  ← 90ms 处还有明显能量
                  50:0.0185 40:0.0144 30:0.0131 20:0.0102 10:0.0068 0:0.0032  ← 被削平
```

现在改成**智能淡出**（`smart_fade_out`）：先判断末尾是否已经自然收束，是则完全不动；
否则只对最后那点残留能量做淡出，并把它之前的样本清零，避免削到有效发声。

```
自然收束的音频:  末样本/峰值=0.0000  →  最大改动=0.00000  (完全不碰)
仍在发声的音频:  末样本/峰值=0.8000  →  只淡出最后的残留部分
```

这样末字既能完整说完，末尾又不会有突兀的 click 声。最后再补 **120ms 尾部静音**。

#### 附带修复：短文本整段丢失

原来前导静音裁剪需要等满 0.6s（14400 采样点）才能判断，而**整段音频比这还短时**
（例如 `你好` 只有 ~330ms），循环结束时缓存里的音频从未被输出，导致**语音整段丢失**。

现在在收尾处补上兜底逻辑，把缓存中的音频正常输出：

| 文本 | 修复前 | 修复后 |
| --- | --- | --- |
| `你好` | 2880 samples，**纯静音** | 9621 samples，有语音 |
| `测试` | 2880 samples，**纯静音** | 10282 samples，有语音 |

## 实现说明

- **并发模型**：ChatTTS 的 `Chat` 实例非线程安全且推理是重 GPU 操作，因此服务端使用
  **全局单队列 + 单 worker**（`examples/ws_server/main.py` 的 `worker_loop`），所有客户端
  的请求排队逐个合成。
- **流式**：服务端固定 `split_text=False`，并把 `stream_batch` 调到 8、`pass_first_n_batches`
  设为 0，避免 ChatTTS 默认丢弃前两批输出导致短句没有音频。
- **PCM 转换**：使用固定满量程缩放（`examples/ws_server/engine.py` 的 `float_to_pcm16`），
  而不是 `tools/audio/np.py` 的逐块按最大值归一化——后者在分块流式下会造成块间音量跳变。
- **打断**：`stop` / `cancel` 调用 `Chat.interrupt()`，通过 `GPT.Context` 标志位让当前生成提前结束。
- **文本规范化**：`en` 使用 `nemo_text_processing`，`zh` 需要 `WeTextProcessing`；缺失时只告警，
  不阻塞服务启动。
- **本地播报**（`examples/ws_server/player.py`）：PCM 走 `aplay` 子进程的标准输入，
  边生成边写入，实现低延迟出声；MP3/WAV 先写临时文件再用 `ffplay` 播放。
  播报器只在单 worker 线程里使用，不存在并发问题。
- **音频后处理**（`examples/ws_server/audio.py`）：裁前导静音、保音高变速（相位声码器）、
  智能淡出与补静音。全部在流式下逐块进行，不影响首包延迟。
- **末字完整性**（`examples/ws_server/engine.py`）：自动补终止标点 + 检测到硬切时换种子重试，
  见上文「结尾突兀」。可用 `TTS_AUTO_PUNCTUATE=0` 关闭自动补标点。

### 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TTS_HOST` | `0.0.0.0` | 监听地址 |
| `TTS_PORT` | `8000` | 监听端口 |
| `TTS_PLAY` | `0` | 设为 `1` 时服务端默认本地播报 |
| `TTS_AUTO_PUNCTUATE` | `1` | 设为 `0` 时不自动补终止标点（关闭末字完整性优化） |
