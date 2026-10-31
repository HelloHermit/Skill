# 已知坑与解法

按踩坑概率排序。遇到新坑，补在最上面。

## 1. RTX 50 系（Blackwell, sm_120）int8 直接崩

现象：`RuntimeError: cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED`，或
`ValueError: Requested int8_float16 compute type, but the target device ... do not support`。

原因：CTranslate2 的 `auto` 在 sm_120 上会挑 int8/int8_float16，而该路径对维度不整除 4 的
GEMM（Whisper 词表 51865）不支持。

解法：显式 `--compute float16`。不要用 `auto`，也不要指望升级包能自动避开。

## 2. pip 版 ctranslate2 不带 CUDA 运行时

现象：`Library cublas64_12.dll is not found or cannot be loaded`（模型能加载，一编码就炸）。

解法：

```bash
uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

并让进程能找到 DLL（`scripts/transcribe.py` 已内置）：

```python
import glob, os, sys
for d in glob.glob(os.path.join(sys.prefix, "Lib/site-packages/nvidia/*/bin")):
    os.add_dll_directory(d)
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
```

Linux 对应 `LD_LIBRARY_PATH`，路径是 `.../nvidia/*/lib`。

## 3. 模型下载卡在 0.3 MB/s

现象：15 分钟只下了 259 MB，看起来"卡死"其实在爬。

解法：`export HF_ENDPOINT=https://hf-mirror.com`。判断是否真卡：比对 `du -sh models` 两次。

## 4. 批处理生成器被提前消费

现象：`BatchedInferencePipeline.transcribe()` 只产出 1 段就结束，日志显示
`segments=1 total=100s`（2.7 小时音频 100 秒跑完，明显不对）。

原因：把 generator 装了 box（`segments = [first]`），后续迭代就只剩那一个。

解法：`from itertools import chain; segments = chain([first], it)`。

## 5. 长音频的幻觉片头

现象：开头出现"请不吝点赞、订阅、转发、打赏支持明镜与点点栏目"之类与内容无关的句子。

原因：Whisper 训练数据里的字幕组残留，在静音/音乐段被激活。

解法：`vad_filter=True` + `condition_on_previous_text=False`，后处理再用正则过滤一次
（`scripts/postprocess.py` 的 `DEFAULT_HALLUCINATIONS`）。

## 6. 最后几分钟内容丢失

现象：转写比视频短一截。

排查：`tail -3 *.txt` 看最后时间码是否接近总时长；多半是 VAD 把结尾长静音连同语音一起切了，
或 ffmpeg 抽音频时被中断。重抽音频并确认 `du -h` 与时长匹配。

## 7. ffmpeg showinfo 在 `-v error` 下没有输出

现象：想用 `select='gt(scene,0.1)',showinfo` 做转场检测，结果一行都没有。

原因：showinfo 打在 INFO 级别，被 `-v error` 吞了。

解法：用 `-hide_banner -loglevel info`，别用 `-v error`。或者干脆放弃转场检测，改成固定间隔抽帧 +
文字相似度聚类（`scripts/ocr_frames.py`），对讲座类视频更稳。

## 8. Windows 上中文路径 / 长路径

现象：ffmpeg 或 Python 报文件不存在。

解法：工作目录用纯 ASCII（`work/`），源文件路径加引号传参；不要在处理过程中重命名源文件。

## 9. 不给 initial_prompt，输出几乎没有标点

现象：整段转写只有逗号、没有句号，`postprocess.py` 合并后变成一坨几百字的长句。

原因：Whisper 的标点行为受 `initial_prompt` 影响，空 prompt 时中文标点会退化成极简。

解法：始终传一段**本身带标点的中文**作为 prompt，例如"以下是一门关于……的课程录音，使用简体中文。"。
这段文字同时承担词表引导与标点示范两个作用。

## 10. OCR 把桌面杂物当幻灯片

现象：抽帧 OCR 结果里全是任务栏、WPS 菜单、聊天窗口。

解法：`ocr_frames.py` 里的 `JUNK` 正则先过滤 UI 文案；再按文字相似度聚类，只保留稳定出现的
整屏文本。判断标准：同一段文字连续 3 帧以上不变，才算一张幻灯片。
