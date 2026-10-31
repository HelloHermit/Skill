#!/usr/bin/env python
"""OCR sampled video frames, then group them into slide segments.

Purpose is cross-checking, not transcription: slides usually carry the correct
spelling of names/terms that ASR mangles, plus bullet points the speaker never
read out loud.

Requires: pip install rapidocr-onnxruntime "numpy<2" pillow
"""
import argparse
import difflib
import glob
import json
import os
import re

JUNK = re.compile(
    r"99\+|WPS|稻壳|月\d{1,2}日|\d{1,2}:\d{2}|直播|素材|升级|文件|开始|插入|"
    r"设计|切换|放映|审阅|视图|工具|会员|三文件|个人中心|互动区"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("frames_dir")
    p.add_argument("out_jsonl")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between frames")
    p.add_argument("--slides", default=None, help="also write grouped slide segments here")
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--similarity", type=float, default=0.72, help="below this = new slide")
    args = p.parse_args()

    from rapidocr_onnxruntime import RapidOCR  # imported late: heavy
    engine = RapidOCR()

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "f_*.jpg")))
    if not frames:
        frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frames:
        raise SystemExit(f"no frames found in {args.frames_dir}")

    prev, slides, cur = None, [], None
    with open(args.out_jsonl, "w", encoding="utf-8") as out:
        for i, fp in enumerate(frames):
            t = int(i * args.interval)
            res, _ = engine(fp)
            lines = []
            if res:
                for box, txt, score in res:
                    txt = txt.strip()
                    if score and score > args.min_score and len(txt) >= 2 and not JUNK.search(txt):
                        lines.append((box[0][1], txt))
            lines.sort(key=lambda x: x[0])
            text = "\n".join(t for _, t in lines)

            out.write(json.dumps({"i": i, "t": t, "text": text}, ensure_ascii=False) + "\n")
            out.flush()

            if prev is None or difflib.SequenceMatcher(None, prev, text).ratio() < args.similarity:
                if cur is not None:
                    slides.append(cur)
                cur = {"start": t, "text": text}
            elif text and len(text) > len(cur.get("text", "")):
                cur["text"] = text
            prev = text

            if i % 50 == 0:
                print(f"[ocr] {i}/{len(frames)} t={t}s", flush=True)

    if cur is not None:
        slides.append(cur)

    if args.slides:
        with open(args.slides, "w", encoding="utf-8") as f:
            json.dump(slides, f, ensure_ascii=False, indent=1)

    print(f"[done] frames={len(frames)} slide_segments={len(slides)}")


if __name__ == "__main__":
    main()
