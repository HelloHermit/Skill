#!/usr/bin/env bash
# Probe a media file and the local ASR environment before starting a long job.
# usage: bash probe.sh <video-or-audio>
set -uo pipefail

IN="${1:-}"
if [ -z "$IN" ] || [ ! -f "$IN" ]; then
  echo "usage: bash probe.sh <video-or-audio>" >&2
  exit 1
fi

echo "== file =="
ls -lh "$IN" | awk '{print $5, $9}'

echo
echo "== streams =="
ffprobe -v error \
  -show_entries format=duration,size,bit_rate \
  -show_entries stream=index,codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,bit_rate \
  -of default=noprint_wrappers=0 "$IN"

echo
echo "== duration (human) =="
ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN" \
  | awk '{d=$1; printf "%d:%02d:%02d\n", d/3600, (d%3600)/60, d%60}'

echo
echo "== tools =="
ffmpeg -version 2>/dev/null | head -1 || echo "ffmpeg: MISSING"
python --version 2>/dev/null || echo "python: MISSING"

echo
echo "== gpu =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no nvidia-smi (CPU-only path)"
fi

echo
echo "== guidance =="
echo "- 音频采样率 <16000 或码率 <64kbps: 转写质量上限受限, 先在 reports 里说明"
echo "- 无 GPU: 用 whisper.cpp 或 medium/int8, 预期时间约为音频时长的 0.3-1x"
echo "- 有 GPU (>=6GB): faster-whisper large-v3 + float16, 预期 6-16x 实时"
