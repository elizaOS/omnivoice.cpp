#!/usr/bin/env python3
"""Cross decode probe : feed CPP mg-tokens to the Python audio decoder.

If decoder_pt(tokens_cpp) matches tts-cpp.wav, the C++ decoder is fine and
the divergence lives entirely in the LM/MaskGIT path. If it does not match,
the C++ audio decoder has a bug too.

We also feed PT tokens to the same Python decoder as a sanity reference,
which must reproduce tts-python.wav (modulo whatever post we still have).
"""

import os
import struct

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice
from omnivoice.utils.common import fix_random_seed

CKPT = "../checkpoints/OmniVoice"


def load_dump(path):
    raw = np.fromfile(path, dtype=np.uint8)
    nd = int(np.frombuffer(raw[0:4], dtype=np.int32)[0])
    sh = tuple(int(x) for x in np.frombuffer(raw[4:4 + 4 * nd], dtype=np.int32))
    body = np.frombuffer(raw[4 + 4 * nd:], dtype=np.float32)
    return body.reshape(sh)


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0


def decode_tokens(model, tokens_kt):
    """tokens_kt is [K, T] int. Returns float32 mono numpy array of samples."""
    device = next(model.parameters()).device
    t = torch.from_numpy(tokens_kt.astype(np.int64)).to(device)
    t = t.unsqueeze(0)
    with torch.no_grad():
        out = model.audio_tokenizer.decode(t)
    wav = getattr(out, "audio_values", out)
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().to(torch.float32).cpu().numpy()
    if wav.ndim == 3:
        wav = wav[0, 0]
    elif wav.ndim == 2:
        wav = wav[0]
    return np.asarray(wav, dtype=np.float32)


def main():
    fix_random_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(
        CKPT,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()

    tok_cpp = load_dump("cpp/mg-tokens.bin").astype(np.int32)
    tok_pt  = load_dump("python/mg-tokens.bin").astype(np.int32)
    print(f"[Input] Tokens ggml: {tok_cpp.shape} python: {tok_pt.shape}")

    audio_pt_from_cpp = decode_tokens(model, tok_cpp)
    audio_pt_from_pt  = decode_tokens(model, tok_pt)

    sf.write("python/decode-of-cpp-tokens.wav", audio_pt_from_cpp, 24000, subtype="FLOAT")
    sf.write("python/decode-of-pt-tokens.wav",  audio_pt_from_pt,  24000, subtype="FLOAT")

    cpp_wav, _ = sf.read("cpp/tts-cpp.wav")
    pt_wav,  _ = sf.read("python/tts-python.wav")
    if cpp_wav.ndim > 1:
        cpp_wav = cpp_wav[:, 0]
    if pt_wav.ndim > 1:
        pt_wav = pt_wav[:, 0]
    cpp_wav = cpp_wav.astype(np.float32)
    pt_wav  = pt_wav.astype(np.float32)

    raw_cpp = load_dump("cpp/output-audio.bin").astype(np.float32)
    raw_pt  = load_dump("python/output-audio.bin").astype(np.float32)

    # Sanity: the Python decoder on Python tokens must reproduce the Python
    # raw audio and the Python WAV. Anything below ~1.0 here means the
    # reference path itself is unstable.
    print(f"[Sanity] PyDecoder(PyTokens) vs PyRaw: {cos(audio_pt_from_pt, raw_pt):.6f}")
    print(f"[Sanity] PyDecoder(PyTokens) vs PyWAV: {cos(audio_pt_from_pt, pt_wav):.6f}")

    # Cross: the Python decoder fed the GGML tokens. If this matches the GGML
    # raw audio, the GGML decoder is bit equivalent and the divergence lives
    # entirely upstream in the LM and MaskGIT path.
    print(f"[Cross] PyDecoder(GgmlTokens) vs GgmlRaw: {cos(audio_pt_from_cpp, raw_cpp):.6f}")
    print(f"[Cross] PyDecoder(GgmlTokens) vs GgmlWAV: {cos(audio_pt_from_cpp, cpp_wav):.6f}")
    print(f"[Cross] PyDecoder(GgmlTokens) vs PyWAV: {cos(audio_pt_from_cpp, pt_wav):.6f}")

    n = min(tok_cpp.size, tok_pt.size)
    a = tok_cpp.ravel()[:n]
    b = tok_pt.ravel()[:n]
    print(f"[Cossim] Tokens exact: {100.0 * float((a == b).mean()):.2f}%")


if __name__ == "__main__":
    main()
