#!/usr/bin/env python
"""Transcribe long audio with faster-whisper and emit .srt / .jsonl / .txt.

Design notes (do not "simplify" these away):
- compute_type defaults to float16: on RTX 50-series (Blackwell, sm_120) the
  int8 / int8_float16 paths abort with CUBLAS_STATUS_NOT_SUPPORTED.
- VAD is on by default: long audio without VAD drifts into hallucinated
  boilerplate ("请不吝点赞订阅转发打赏...").
- The batched pipeline returns a generator; consuming it into a list before
  iterating yields exactly one segment. Keep the chain().
"""
import argparse
import glob
import json
import os
import sys
import time
from itertools import chain


def add_cuda_dll_dirs():
    """Make pip-installed CUDA runtime DLLs visible to CTranslate2 (Windows)."""
    if not hasattr(os, "add_dll_directory"):
        return
    roots = [
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia"),
        os.path.join(sys.prefix, "lib", "site-packages", "nvidia"),
    ]
    for root in roots:
        for sub in ("bin", "lib"):
            for d in glob.glob(os.path.join(root, "*", sub)):
                try:
                    os.add_dll_directory(d)
                    os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                except OSError:
                    pass


def fmt(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("out_prefix", help="e.g. work/lecture -> work/lecture.srt")
    p.add_argument("--model", default="large-v3")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--compute", default="float16")
    p.add_argument("--batch", type=int, default=8, help="0 = sequential, no batching")
    p.add_argument("--lang", default="zh")
    p.add_argument("--beam", type=int, default=5)
    p.add_argument("--prompt", default=None, help="initial_prompt: domain vocabulary")
    p.add_argument("--no-vad", action="store_true")
    p.add_argument("--model-dir", default=None)
    args = p.parse_args()

    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    if args.device == "cpu" and args.compute == "float16":
        args.compute = "int8"

    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute,
        download_root=args.model_dir or os.path.join(os.getcwd(), "models"),
    )
    print(f"[load] {time.time()-t0:.1f}s model={args.model} device={args.device} compute={args.compute}",
          file=sys.stderr, flush=True)

    kwargs = dict(
        language=args.lang,
        beam_size=args.beam,
        vad_filter=not args.no_vad,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
        condition_on_previous_text=False,
        initial_prompt=args.prompt,
    )

    t1 = time.time()
    if args.batch > 0 and args.device == "cuda":
        try:
            pipe = BatchedInferencePipeline(model=model)
            segs, info = pipe.transcribe(args.audio, batch_size=args.batch, **kwargs)
            it = iter(segs)
            first = next(it)
            print(f"[batched] first segment in {time.time()-t1:.1f}s", file=sys.stderr, flush=True)
            segments = chain([first], it)
        except Exception as exc:  # noqa: BLE001 - fall back rather than lose the job
            print(f"[warn] batched failed ({exc}); sequential fallback", file=sys.stderr, flush=True)
            t1 = time.time()
            segments, info = model.transcribe(args.audio, **kwargs)
    else:
        segments, info = model.transcribe(args.audio, **kwargs)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    n = 0
    with open(args.out_prefix + ".srt", "w", encoding="utf-8") as fsrt, \
            open(args.out_prefix + ".txt", "w", encoding="utf-8") as ftxt, \
            open(args.out_prefix + ".jsonl", "w", encoding="utf-8") as fjson:
        for seg in segments:
            n += 1
            text = seg.text.strip()
            fsrt.write(f"{n}\n{fmt(seg.start)} --> {fmt(seg.end)}\n{text}\n\n")
            ftxt.write(f"[{fmt(seg.start)}] {text}\n")
            fjson.write(json.dumps({"i": n, "start": round(seg.start, 2),
                                    "end": round(seg.end, 2), "text": text},
                                   ensure_ascii=False) + "\n")
            if n % 50 == 0:
                el = time.time() - t1
                print(f"[{n} segs] audio_at={fmt(seg.end)} elapsed={el:.0f}s "
                      f"rtf={el/max(seg.end,1):.3f}", file=sys.stderr, flush=True)

    print(f"[done] segments={n} total={time.time()-t0:.0f}s dur={fmt(info.duration)}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
