#!/usr/bin/env bash
# Extract a normalized 16 kHz mono PCM track for ASR.
# usage: bash extract_audio.sh <video> <out.wav>
set -euo pipefail

IN="${1:-}"
OUT="${2:-work/audio_16k.wav}"
if [ -z "$IN" ] || [ ! -f "$IN" ]; then
  echo "usage: bash extract_audio.sh <video> [out.wav]" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

start=$(date +%s)
ffmpeg -hide_banner -loglevel error -stats \
  -i "$IN" -vn -ac 1 -ar 16000 -c:a pcm_s16le \
  -y "$OUT"
end=$(date +%s)

echo "audio -> $OUT ($(du -h "$OUT" | cut -f1), $((end-start))s)"
