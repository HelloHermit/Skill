# ASR 方案选型

数据为 2026-10 实测（GitHub API + 本机 RTX 5060 Laptop 8 GB / Windows / Python 3.11）。星数与提交时间会变，判断逻辑不变。

## 四个主流开源方案

| 项目 | Stars | 许可证 | 引擎 | 中文表现 | 适用边界 |
|---|---|---|---|---|---|
| **SYSTRAN/faster-whisper** | ~26k | MIT | CTranslate2 重实现 Whisper | 好（专名弱） | 有 GPU、要速度与 VAD、要词级时间戳。**默认首选** |
| openai/whisper | ~112k | MIT | PyTorch 原版 | 同上 | 只在需要与官方实现严格对齐时用；速度慢 3–4 倍，无内置 VAD |
| ggml-org/whisper.cpp | ~55k | MIT | C/C++ + 量化 | 同上 | 无 GPU、边缘设备、需单文件分发、内存受限 |
| modelscope/FunASR | ~21k | MIT | Paraformer / SenseVoice | **最好**（含标点、说话人） | 中文会议、多人对话、需要说话人分离；依赖 torch + ModelScope，环境更重 |

结论分档：

- 默认：faster-whisper + large-v3 + float16 + batch。中文讲座可读性够用，专名靠 OCR/词表兜。
- 只有 CPU：whisper.cpp（量化 medium）或 faster-whisper int8 + small/medium，先跟用户交代时间成本。
- 中文强需求 / 要区分说话人：FunASR。它自带 VAD、标点、CAM++ 说话人，省掉后处理，但 Windows + 新卡上 torch 依赖链更容易出问题。
- 只要"能看懂大意"且不在乎隐私：商用 API 最快，但长音频要切片（多数限制 25 MB / 单文件），且逐段请求会丢上下文。

## 本机实测基线（可直接写给用户看）

| 指标 | 数值 |
|---|---|
| 音频 | 162 分 40 秒 / 16 kHz 单声道 PCM |
| 模型 | large-v3，float16，batch_size=8 |
| 模型加载 | 18–27 秒（本地缓存） |
| 转写总耗时 | 10 分 25 秒 |
| RTF | ≈0.064（约 15.6× 实时） |
| 产出 | 345 段，42,539 字 |
| 显存占用 | <6 GB（8 GB 卡有余量） |

同机 CPU 路线（int8 large-v3，8 核）经验值约 0.3–0.5× 实时，同样是 2.7 小时音频要 5–9 小时——**这就是为什么值得先修 GPU 依赖，而不是降级跑**。

## 决策树

```
有 NVIDIA GPU (>=6GB)?
├─ 是 → 需要说话人分离?
│        ├─ 是 → FunASR
│        └─ 否 → faster-whisper (large-v3, float16, batch 4-8)
└─ 否 → 时长 > 1 小时?
         ├─ 是 → whisper.cpp 量化 medium（可接受精度换时间）
         └─ 否 → faster-whisper CPU int8 + small/medium
```

## 模型下载

HuggingFace 直连在国内经常只有 0.3 MB/s（3 GB 模型要 2 小时以上）。用镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

实测 hf-mirror 约 7.7 MB/s，ModelScope 源约 13 MB/s。模型只需下一次，缓存在 `download_root` 指定目录。

## 精度预期（别向用户承诺"逐字准确"）

- 普通话讲座：语义完整、断句合理、标点基本正确
- 主要错误：**人名、机构名、产品名、版本号**（同音替换），以及数字单位
- 次要错误：口语重复词、语气词
- 交付时必须声明：**专名以 OCR 或用户确认为准**，并在无法确认处标"（音）"
