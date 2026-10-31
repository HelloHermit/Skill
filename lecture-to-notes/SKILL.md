---
name: lecture-to-notes
description: 把长视频、录播课、会议录音或播客转成可用的结构化笔记：抽取音频、GPU 语音识别、抽帧 OCR 校验、按主题分层总结，并产出带时间码的字幕。当用户提供 mp4/mkv/mov/wav/m4a/mp3 等音视频文件并要求"总结这个视频/转写/提取字幕/整理课程笔记/做会议纪要"，或需要给 1 小时以上的长音频建立可检索文字存档时使用。
agent_created: true
---

# Lecture to Notes

## Overview

把"一个 2 小时的视频"变成"一份能直接用的结构化笔记 + 一套可检索字幕"。核心不是转写本身，而是三件事：**音频质量先过关、专有名词必须交叉校验、总结必须带时间码锚点**。转写占比不到一半工作量。

## 适用与不适用

适用：
- 录播课、直播回放、学术讲座、内部培训视频（30 分钟以上收益明显）
- 会议录音、访谈、播客（需要分主题纪要或逐段摘录）
- 已有视频需要字幕文件（SRT）

不适用（不要硬上）：
- 纯音乐、强噪声、多人抢话无麦克风分离 → 先做降噪或要求原始音轨，否则转写不可用
- 只有 5 分钟、只想要一句话结论 → 直接看，不值得起流水线
- 涉密内容 → 不要送云端 API，全程本地（本 skill 默认本地）

## 五步流程

### Step 1 探测（30 秒，必做）

```bash
bash scripts/probe.sh "<input_video>"
```

输出时长、音视频流参数、GPU 型号/显存、ffmpeg 与 python 版本。据此决定：
- 音频采样率 < 16 kHz 或码率 < 64 kbps → 先提示用户音质可能是转写质量上限
- 无 NVIDIA GPU → 见 `references/asr-selection.md` 的 CPU 路线（whisper.cpp 或 small 模型），并把预期时间告诉用户

### Step 2 抽音频

```bash
bash scripts/extract_audio.sh "<input_video>" work/audio_16k.wav
```

统一转 16 kHz 单声道 PCM。不要直接喂 mp4 给识别器：容器解码的不确定性会拖慢并引入噪声。2.7 GB / 2h42m 的视频约 17 秒完成，输出约 300 MB。

### Step 3 语音识别

```bash
python scripts/transcribe.py work/audio_16k.wav work/lecture \
  --model large-v3 --device cuda --compute float16 --batch 8 --lang zh \
  --prompt "以下是一门关于……的课程录音，使用简体中文。"
```

产出 `work/lecture.srt`、`work/lecture.jsonl`（带起止秒）、`work/lecture.txt`。

关键默认值（已写进脚本，改动前先读 `references/troubleshooting.md`）：
- `--compute float16`：RTX 50 系（Blackwell, sm_120）上 int8 会直接崩，不要用 `auto`
- `--batch 8`：批处理推理，消费级 8 GB 显存的安全值；OOM 就降到 4
- VAD 默认开启：长音频不开 VAD 会产出"请不吝点赞订阅"这类幻觉
- `initial_prompt` 必传：既降低专名错误，也决定标点质量。不传时中文输出常常没有句末标点，段落合并会粘成一片（实测同一段音频，传与不传的可读性差一个档）

### Step 4 可选：抽帧 OCR 交叉校验

视频里出现幻灯片、板书、屏幕共享时做，纯口播可跳过。

```bash
ffmpeg -hide_banner -loglevel error -i "<input_video>" -vf "fps=1/10,scale=1280:-1" -q:v 4 work/frames/f_%05d.jpg
python scripts/ocr_frames.py work/frames work/ocr.jsonl --interval 10 --slides work/slides.json
```

作用有两个，第二个更重要：
1. 提取幻灯片文字，补上口播没念出来的要点
2. **校验专有名词**——ASR 最容易错的就是人名、产品名、机构名，而幻灯片上通常写着正确写法

### Step 5 结构化写作

先跑后处理，把碎句合成段落：

```bash
python scripts/postprocess.py work/lecture.jsonl work/transcript.md
```

然后按 `assets/summary-template.md` 的结构写正文。写作硬性要求：

1. **每个论断挂时间码**，形如 `（01:18–02:20）`，让人能回去考古原话
2. **区分"原话"与"归纳"**：引用加引号并保留口语特征，归纳要标明是归纳
3. **必须有批判节**：哪些说法有证据、哪些是幸存者偏差、哪些与常识冲突
4. **必须有落地节**：把内容翻译成读者的具体动作，否则总结没有价值
5. 专名以 OCR/用户确认为准，ASR 结果只作参考；不确定的标注"（音）"

## 选型速查

| 场景 | 选择 | 理由 |
|---|---|---|
| 有 NVIDIA GPU，要质量与速度 | faster-whisper + large-v3 + float16 | 本机实测 15.6× 实时，内置 VAD |
| 无 GPU，CPU 长音频 | whisper.cpp（量化）+ medium | 可移植，边缘设备友好 |
| 中文为主、要说话人分离 | FunASR（Paraformer/SenseVoice） | 自带 VAD+标点+说话人，依赖链更重 |
| 一次性、可上云、不想装环境 | 商用 ASR API 或平台内置识别 | 省事，但长音频要分片、有隐私成本 |

详细对比与实测数据见 `references/asr-selection.md`。

## 交付前自检

- [ ] 音频时长与视频时长一致（抽音频没截断）
- [ ] 转写末尾有内容（长音频常见"最后 5 分钟丢失"）
- [ ] 抽 2–3 分钟片段人耳比对，确认不是整段错位
- [ ] 人名/机构名已用 OCR 或用户确认校准
- [ ] 时间码抽查 3 处，能对上原片
- [ ] 产物命名：英文 snake_case，正文与字幕分开

## 资源

- `scripts/probe.sh` — 探测时长/流参数/GPU
- `scripts/extract_audio.sh` — 抽 16 kHz 单声道
- `scripts/transcribe.py` — faster-whisper 转写，输出 srt/jsonl/txt
- `scripts/postprocess.py` — 去幻觉、合段落、生成带时间码的 md
- `scripts/ocr_frames.py` — 抽帧 OCR + 幻灯片聚类
- `references/asr-selection.md` — 四个开源方案对比与适用边界
- `references/troubleshooting.md` — 已知坑与解法（GPU/依赖/下载/生成器）
- `assets/summary-template.md` — 结构化总结模板（概览→时间轴→分主题→工具→批判→落地→技术附录）
