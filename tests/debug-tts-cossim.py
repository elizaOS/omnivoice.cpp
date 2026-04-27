#!/usr/bin/env python3
"""Cossim debug : C++ omnivoice-tts vs Python OmniVoice on voice design.

Inputs (relative to CWD = tests/) :
    prompt.txt       target text fed to both pipelines

Both sides run with :
    instruct=male, language=French, seed=42, F32 weights, no pre or post
    process. Defaults match : num_step=32, guidance_scale=2.0, t_shift=0.1,
    layer_penalty_factor=5.0, position_temperature=5.0, class_temperature=0.0.

Dumps land in cpp/ (C++) and python/ (Python). The script compares each
matching .bin pair via cosine similarity over the f32 payload, plus exact
match rate for tensors that originated as int (mg-tokens). All paths are
relative, no absolute paths anywhere.
"""

import argparse
import os
import struct
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch

# Enforce strict F32 math on the Python reference path. PyTorch defaults
# allow F16 / BF16 reduction inside F32 matmul and TF32 inside cudnn even
# when matmul.allow_tf32 is False, which silently drifts the reference.
torch.backends.cuda.matmul.allow_tf32                            = False
torch.backends.cudnn.allow_tf32                                  = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.set_float32_matmul_precision("highest")

from omnivoice import OmniVoice
from omnivoice.utils.common import fix_random_seed

BIN        = "../build/omnivoice-tts"
MODEL_LM   = "../models/omnivoice-base-F32.gguf"
MODEL_CDC  = "../models/omnivoice-tokenizer-F32.gguf"
CKPT       = "../checkpoints/OmniVoice"
DUMP_CPP   = "cpp"
DUMP_PT    = "python"

def cuda_props():
    """Return (sm_count, max_threads_per_sm) of the active CUDA device.
    Used to mirror PyTorch's calc_execution_policy in the C++ Philox helper.
    Returns (0, 0) when CUDA is unavailable, which is fine for tests run on
    CPU (the C++ path falls back to a single Philox block per kernel)."""
    if not torch.cuda.is_available():
        return 0, 0
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return p.multi_processor_count, p.max_threads_per_multi_processor

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_dump(path, data):
    """Write a tensor in the C++ debug.h format :
        [ndim:i32] [shape:i32 x ndim] [data:f32 x numel]
    """
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
    """Inverse of save_dump : returns (data:f32 numpy, shape:tuple)."""
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

    # First call to _prepare_embed_inputs at step 0 returns the input embedding
    # right before layer 0 of the LLM, mirroring the C++ inputs_embeds dump.
    # Also captures the raw input_ids row k=0 for cond and uncond, so any
    # token-level divergence localizes upstream of the embed lookup.
    seen_embed = {"done": False}
    orig_prepare = model._prepare_embed_inputs
    def hooked_prepare(input_ids, audio_mask):
        out = orig_prepare(input_ids, audio_mask)
        if not seen_embed["done"] and out.dim() == 3 and out.shape[0] >= 2:
            cond   = out[0].detach().to(torch.float32).cpu().numpy()
            uncond = out[1].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "lm-hidden-step0-cond-embed.bin"),   cond)
            save_dump(os.path.join(dump_dir, "lm-hidden-step0-uncond-embed.bin"), uncond)
            # Style and text tokens duplicate across all K codebooks so k=0
            # carries the full sequence for diagnostic. Cast to f32 keeps the
            # debug.h binary format identical on both sides.
            cond_ids   = input_ids[0, 0, :].detach().to(torch.float32).cpu().numpy()
            uncond_ids = input_ids[1, 0, :].detach().to(torch.float32).cpu().numpy()
            save_dump(os.path.join(dump_dir, "prompt-cond-ids.bin"),   cond_ids)
            save_dump(os.path.join(dump_dir, "prompt-uncond-ids.bin"), uncond_ids)
            seen_embed["done"] = True
        return out
    model._prepare_embed_inputs = hooked_prepare

    # Bisection : dump cond and uncond hidden states after a few layers so a
    # mismatch can be localized within the 28 layer Qwen3 stack.
    bisect_layers = [0, 6, 13, 20]
    seen_layers   = {idx: False for idx in bisect_layers}
    def make_layer_hook(layer_idx):
        def hook(module, inputs, output):
            if seen_layers[layer_idx]:
                return
            h = output[0] if isinstance(output, tuple) else output
            if h.dim() == 3 and h.shape[0] >= 2:
                cond   = h[0].detach().to(torch.float32).cpu().numpy()
                uncond = h[1].detach().to(torch.float32).cpu().numpy()
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-cond-l{layer_idx}.bin"),   cond)
                save_dump(os.path.join(dump_dir, f"lm-hidden-step0-uncond-l{layer_idx}.bin"), uncond)
                seen_layers[layer_idx] = True
        return hook
    for layer_idx in bisect_layers:
        model.llm.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))

    # First call to audio_heads corresponds to step 0 of the MaskGIT loop.
    # The input to audio_heads is the final hidden state, shape [B, S, D],
    # mirroring what the C++ side reads back via dump_hidden_dir before the
    # lm_head matmul. We dump cond (b=0) and uncond (b=1) separately.
    seen_hidden = {"done": False}
    def pre_audio_heads(module, inputs):
        if not seen_hidden["done"]:
            h = inputs[0]
            if h.dim() == 3 and h.shape[0] >= 2:
                cond   = h[0].detach().to(torch.float32).cpu().numpy()
                uncond = h[1].detach().to(torch.float32).cpu().numpy()
                save_dump(os.path.join(dump_dir, "lm-hidden-step0-cond.bin"),   cond)
                save_dump(os.path.join(dump_dir, "lm-hidden-step0-uncond.bin"), uncond)
                seen_hidden["done"] = True
    model.audio_heads.register_forward_pre_hook(pre_audio_heads)

    # First call to _predict_tokens_with_scoring corresponds to step 0 of the
    # MaskGIT loop. Capture cond and uncond logits in [K, T, V] layout (squeeze
    # the batch axis to match the C++ dump shape).
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
        # out is a list of (K, T_i) long tensors, one per batch item.
        save_dump(os.path.join(dump_dir, "mg-tokens.bin"), out[0])
        return out
    model._generate_iterative = hooked_generate

    orig_decode = model.audio_tokenizer.decode
    def hooked_decode(*args, **kwargs):
        out = orig_decode(*args, **kwargs)
        # The audio tokenizer returns either a tensor or a wrapper holding
        # audio_values shape [B, C, N]. Unwrap and dump the first item mono.
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
    ap.add_argument("--prompt",   default="prompt.txt")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--instruct", default="male")
    ap.add_argument("--lang",     default="French")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--pos-temp", type=float, default=None,
                    help="Override MaskGIT position_temperature on both sides (default 5.0)")
    ap.add_argument("--cls-temp", type=float, default=None,
                    help="Override MaskGIT class_temperature on both sides (default 0.0)")
    ap.add_argument("--out-cpp",  default="cpp/tts-cpp.wav")
    ap.add_argument("--out-pt",   default="python/tts-python.wav")
    args = ap.parse_args()

    ensure_dir(DUMP_CPP)
    ensure_dir(DUMP_PT)
    os.makedirs(os.path.dirname(args.out_cpp) or ".", exist_ok=True)

    with open(args.prompt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    print(f"[Input] Prompt: {len(text)} chars: {text[:60]}{'...' if len(text) > 60 else ''}")
    print(f"[Input] Instruct: {args.instruct}")
    print(f"[Input] Language: {args.lang}")
    print(f"[Input] Seed: {args.seed}")

    # Python reference path : F32, voice design male, no pre or post process.
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
    gen_kwargs = dict(
        text=text,
        language=args.lang,
        instruct=args.instruct,
        duration=args.duration,
        denoise=True,
        preprocess_prompt=False,
        postprocess_output=False,
        audio_chunk_threshold=1e9,
    )
    if args.pos_temp is not None:
        gen_kwargs["position_temperature"] = args.pos_temp
    if args.cls_temp is not None:
        gen_kwargs["class_temperature"] = args.cls_temp
    audios = model.generate(**gen_kwargs)
    audio_pt = np.asarray(audios[0], dtype=np.float32)
    sf.write(args.out_pt, audio_pt, 24000, subtype="FLOAT")
    print(f"[Python] Audio: {audio_pt.shape[0]} samples {audio_pt.shape[0] / 24000:.2f}s -> {args.out_pt}")

    # Free the GPU before launching the C++ binary so it has room to load
    # the F32 GGUFs without fighting for VRAM.
    del model
    torch.cuda.empty_cache()

    # C++ path : same text, same instruct, same seed, F32 GGUF weights,
    # dumps under cpp/.
    cmd = [
        BIN,
        "--model",       MODEL_LM,
        "--codec",       MODEL_CDC,
        "--seed",        str(args.seed),
        "--sm-count",    str(sm),
        "--sm-threads",  str(mt),
        "--instruct",    args.instruct,
        "--lang",        args.lang,
        "--format",      "wav32",
        "--dump",        DUMP_CPP,
        "--no-fa",
        "-o",            args.out_cpp,
    ]
    if args.duration:
        cmd += ["--duration", str(args.duration)]
    if args.pos_temp is not None:
        cmd += ["--pos-temp", str(args.pos_temp)]
    if args.cls_temp is not None:
        cmd += ["--cls-temp", str(args.cls_temp)]
    print(f"[GGML] Cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, input=text, text=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
    audio_cpp, sr = sf.read(args.out_cpp)
    if audio_cpp.ndim > 1:
        audio_cpp = audio_cpp[:, 0]
    audio_cpp = audio_cpp.astype(np.float32)
    print(f"[GGML] Audio: {audio_cpp.shape[0]} samples {sr} Hz {audio_cpp.shape[0] / sr:.2f}s -> {args.out_cpp}")

    # Cossim in pipeline order: prompt-ids -> embed -> l0 -> l6 -> l13 -> l20
    # -> final -> logits -> tokens -> audio. Cond and uncond on the same line
    # so a drift localizes immediately to the originating stage.
    def pair(name):
        a, _ = load_dump(os.path.join(DUMP_CPP, name))
        b, _ = load_dump(os.path.join(DUMP_PT,  name))
        return a, b

    def ids_exact(a, b):
        n = min(a.size, b.size)
        ai = a.astype(np.int64).ravel()[:n]
        bi = b.astype(np.int64).ravel()[:n]
        diffs = np.where(ai != bi)[0]
        return 100.0 * float(np.mean(ai == bi)), diffs, ai, bi

    ca, cb = pair("prompt-cond-ids.bin")
    ua, ub = pair("prompt-uncond-ids.bin")
    cm, cd, cai, cbi = ids_exact(ca, cb)
    um, ud, uai, ubi = ids_exact(ua, ub)
    print(f"[Cossim] PromptIDs cond exact: {cm:.2f}% uncond exact: {um:.2f}%")
    for s in cd[:20]:
        print(f"[Cossim] PromptIDs cond diff at s={s}: ggml={cai[s]} python={cbi[s]}")
    for s in ud[:20]:
        print(f"[Cossim] PromptIDs uncond diff at s={s}: ggml={uai[s]} python={ubi[s]}")

    stages = [
        ("Embed",  "lm-hidden-step0-{}-embed.bin"),
        ("L0",     "lm-hidden-step0-{}-l0.bin"),
        ("L6",     "lm-hidden-step0-{}-l6.bin"),
        ("L13",    "lm-hidden-step0-{}-l13.bin"),
        ("L20",    "lm-hidden-step0-{}-l20.bin"),
        ("Final",  "lm-hidden-step0-{}.bin"),
        ("Logits", "lm-logits-step0-{}.bin"),
    ]
    for label, fmt in stages:
        ca, cb = pair(fmt.format("cond"))
        ua, ub = pair(fmt.format("uncond"))
        print(f"[Cossim] {label} cond: {cos(ca, cb):.6f} uncond: {cos(ua, ub):.6f}")

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
