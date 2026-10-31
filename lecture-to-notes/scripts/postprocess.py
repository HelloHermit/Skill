#!/usr/bin/env python
"""Turn raw ASR segments into a readable, timestamped transcript.

Two jobs that matter for downstream summarisation:
1. Drop Whisper's training-data hallucinations (subtitle credits, "like and
   subscribe" boilerplate). These appear mostly on silence / music.
2. Merge 2-4 second fragments into paragraphs, so the reader (and the model
   writing the summary) sees sentences, not shards.
"""
import argparse
import json
import re
import sys

DEFAULT_HALLUCINATIONS = [
    r"明镜与点点", r"请不吝点赞", r"订阅.*转发.*打赏", r"字幕由.*提供",
    r"字幕组", r"amara\.org", r"感谢观看", r"谢谢观看",
    r"^\s*$",
]

END_PUNCT = "。？！?!."


def fmt(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    p.add_argument("out_md")
    p.add_argument("--gap", type=float, default=1.5, help="seconds of silence that starts a new paragraph")
    p.add_argument("--max-chars", type=int, default=320)
    p.add_argument("--min-chars", type=int, default=180, help="merge past this only on sentence end")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.jsonl, encoding="utf-8")]
    pat = re.compile("|".join(DEFAULT_HALLUCINATIONS))

    segs = []
    for r in rows:
        text = re.sub(r"\s+", "", r["text"].strip())
        if not text or pat.search(text):
            continue
        segs.append({"start": r["start"], "end": r["end"], "text": text})

    if not segs:
        print("[error] every segment was filtered out; check --jsonl and the filter list",
              file=sys.stderr)
        sys.exit(1)

    paras, cur = [], None
    for s in segs:
        if cur is None:
            cur = dict(s)
            continue
        gap = s["start"] - cur["end"]
        sentence_done = len(cur["text"]) > args.min_chars and cur["text"][-1] in END_PUNCT
        if gap > args.gap or len(cur["text"]) > args.max_chars or sentence_done:
            paras.append(cur)
            cur = dict(s)
        else:
            joiner = "" if cur["text"][-1] in "，。、；：？！,.;:" else "，"
            cur["text"] += joiner + s["text"]
            cur["end"] = s["end"]
    if cur:
        paras.append(cur)

    header = (f"# 课程字幕全文（ASR 转写，{fmt(paras[0]['start'])} - {fmt(paras[-1]['end'])}）\n\n"
              "> 自动转写结果，专有名词可能有误；引用前请核对原片时间码。\n\n")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(header)
        for para in paras:
            f.write(f"**[{fmt(para['start'])}]** {para['text']}\n\n")

    chars = sum(len(x["text"]) for x in paras)
    print(f"segments={len(segs)} paras={len(paras)} chars={chars} -> {args.out_md}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
