#!/usr/bin/env python3
"""Cossim debug : C++ omnivoice-tts vs Python OmniVoice on voice cloning.

Inputs (relative to CWD = tests/) :
    prompt.txt       target text fed to both pipelines
    ref-text.txt     transcript of the cloning reference
    ref-audio.wav    cloning reference audio, any rate, any layout

Both sides run with seed=42, F32 weights, language=French, no pre or post
process. The reference audio is resampled to 24 kHz mono inside both
pipelines.

Dumps land in cpp/ (C++) and python/ (Python) and are compared pair by
pair. All paths are relative.
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.utils.common import fix_random_seed

BIN        = "../build/omnivoice-tts"
MODEL_LM   = "../models/omnivoice-base-F32.gguf"
MODEL_CDC  = "../models/omnivoice-tokenizer-F32.gguf"
CKPT       = "../checkpoints/OmniVoice"
DUMP_CPP   = "cpp"
DUMP_PT    = "python"

def cuda_props():
    if not torch.cuda.is_available():
        return 0, 0
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return p.multi_processor_count, p.max_threads_per_multi_processor

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dump(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().to(torch.float32).cpu().numpy()
    data  = np.ascontiguousarray(data.astype(np.float32))
    shape = data.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", len(shape)))
        for s in shape:
            f.write(struct.pack("i", s))
        f.write(data.tobytes())

def load_dump(path):
    raw   = np.fromfile(path, dtype=np.uint8)
    ndim  = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    shape = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * ndim], dtype=np.int32))
    body  = np.frombuffer(raw[4 + 4 * ndim:], dtype=np.float32)
    return body.reshape(shape), shape

def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0

def install_hooks(model, dump_dir):
    def passthrough_post(generated_audio, postprocess_output, ref_rms):
        return generated_audio
    model._post_process_audio = passthrough_post

    seen = {"step0": False}
    orig_pred = model._predict_tokens_with_scoring
    def hooked_pred(c_logits, u_logits, gen_config):
        if not seen["step0"]:
            c = c_logits.detach().to(torch.float32).cpu().numpy()
            u = u_logits.detach().to(torch.float32).cpu().numpy()
            if c.ndim == 4:
                c = c[0]
            if u.ndim == 4:
                u = u[0]
            save_dump(os.path.join(dump_dir, "lm-logits-step0-cond.bin"),   c)
            save_dump(os.path.join(dump_dir, "lm-logits-step0-uncond.bin"), u)
            seen["step0"] = True
        return orig_pred(c_logits, u_logits, gen_config)
    model._predict_tokens_with_scoring = hooked_pred

    orig_generate = model._generate_iterative
    def hooked_generate(task, gen_config):
        out = orig_generate(task, gen_config)
        save_dump(os.path.join(dump_dir, "mg-tokens.bin"), out[0])
        return out
    model._generate_iterative = hooked_generate

    orig_decode = model.audio_tokenizer.decode
    def hooked_decode(*args, **kwargs):
        out = orig_decode(*args, **kwargs)
        wav = getattr(out, "audio_values", out)
        if isinstance(wav, torch.Tensor):
            arr = wav.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0, 0]
        elif arr.ndim == 2:
            arr = arr[0]
        save_dump(os.path.join(dump_dir, "output-audio.bin"), arr)
        return out
    model.audio_tokenizer.decode = hooked_decode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt",    default="prompt.txt")
    ap.add_argument("--ref-text",  default="ref-text.txt")
    ap.add_argument("--ref-audio", default="ref-audio.wav")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--lang",      default="French")
    ap.add_argument("--duration",  type=float, default=None)
    ap.add_argument("--out-cpp",   default="cpp/clone-cpp.wav")
    ap.add_argument("--out-pt",    default="python/clone-python.wav")
    args = ap.parse_args()

    ensure_dir(DUMP_CPP)
    ensure_dir(DUMP_PT)
    os.makedirs(os.path.dirname(args.out_cpp) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    with open(args.ref_text, "r", encoding="utf-8") as f:
        ref_text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] RefText: {len(ref_text)} chars: {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}")
    print(f"[Input] RefWav: {args.ref_audio}")
    print(f"[Input] Language: {args.lang}")
    print(f"[Input] Seed: {args.seed}")

    fix_random_seed(args.seed)
    sm, mt = cuda_props()
    print(f"[Cuda] sm_count: {sm} max_threads_per_sm: {mt}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = OmniVoice.from_pretrained(
        CKPT,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    install_hooks(model, DUMP_PT)
    audios = model.generate(
        text=text,
        language=args.lang,
        ref_text=ref_text,
        ref_audio=args.ref_audio,
        duration=args.duration,
        denoise=True,
        preprocess_prompt=False,
        postprocess_output=False,
        audio_chunk_threshold=1e9,
    )
    audio_pt = np.asarray(audios[0], dtype=np.float32)
    sf.write(args.out_pt, audio_pt, 24000, subtype="FLOAT")
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {audio_pt.shape[0] / 24000:.2f}s -> {args.out_pt}")

    del model
    torch.cuda.empty_cache()

    cmd = [
        BIN,
        "--model",       MODEL_LM,
        "--codec",       MODEL_CDC,
        "--seed",        str(args.seed),
        "--sm-count",    str(sm),
        "--sm-threads",  str(mt),
        "--ref-wav",     args.ref_audio,
        "--ref-text",    args.ref_text,
        "--lang",        args.lang,
        "--format",      "wav32",
        "--dump",        DUMP_CPP,
        "-o",            args.out_cpp,
    ]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, input=text, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    audio_cpp, sr = sf.read(args.out_cpp)
    if audio_cpp.ndim > 1:
        audio_cpp = audio_cpp[:, 0]
    audio_cpp = audio_cpp.astype(np.float32)
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0] / sr:.2f}s -> {args.out_cpp}")

    # Cossim in pipeline order: logits -> tokens -> audio. Cond and uncond
    # on the same line so a drift localizes to the originating stage.
    def pair(name):
        a, _ = load_dump(os.path.join(DUMP_CPP, name))
        b, _ = load_dump(os.path.join(DUMP_PT,  name))
        return a, b

    ca, cb = pair("lm-logits-step0-cond.bin")
    ua, ub = pair("lm-logits-step0-uncond.bin")
    print(f"[Cossim] Logits cond: {cos(ca, cb):.6f} uncond: {cos(ua, ub):.6f}")

    ta, tb = pair("mg-tokens.bin")
    n = min(ta.size, tb.size)
    ai = ta.astype(np.int64).ravel()[:n]
    bi = tb.astype(np.int64).ravel()[:n]
    print(f"[Cossim] Tokens: {cos(ta, tb):.6f} exact: {100.0 * float(np.mean(ai == bi)):.2f}%")

    aa, ab = pair("output-audio.bin")
    print(f"[Cossim] Audio: {cos(aa, ab):.6f}")

    n = min(audio_cpp.size, audio_pt.size)
    print(f"[Cossim] WAV: {cos(audio_cpp[:n], audio_pt[:n]):.6f} samples: {n}")

if __name__ == "__main__":
    main()
