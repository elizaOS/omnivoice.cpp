// pipeline-tts.cpp: TTS pipeline orchestration.
//
// Phase 1 scope: load OmniVoiceLM weights onto the backend. Subsequent phases
// append the custom embed graph, the audio_heads readout, the MaskGIT loop,
// the prompt builder, and the audio_tokenizer decode. Each compute path
// allocates its own ggml_gallocr at call time, mirroring pipeline-codec.

#include "pipeline-tts.h"

#include "bpe.h"
#include "debug.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "maskgit-tts.h"
#include "pipeline-codec.h"
#include "prompt-tts.h"

#include <cstdio>
#include <string>
#include <vector>

bool pipeline_tts_load(PipelineTTS *  pt,
                       const char *   gguf_path,
                       ggml_backend_t backend,
                       bool           has_gpu,
                       bool           use_fa,
                       bool           clamp_fp16) {
    *pt                = {};
    pt->backend        = backend;
    pt->use_flash_attn = use_fa && has_gpu;
    pt->clamp_fp16     = clamp_fp16;

    if (!gf_load(&pt->gguf, gguf_path)) {
        return false;
    }

    // 1 embed_tokens + 1 final_norm + 1 audio_embeddings + 1 audio_heads
    // + 28 layers * 11 tensors max = 312, headroom to 512.
    wctx_init(&pt->wctx, 512);

    if (!omnivoice_lm_load(&pt->lm, pt->gguf, &pt->wctx)) {
        wctx_free(&pt->wctx);
        gf_close(&pt->gguf);
        return false;
    }

    if (!wctx_alloc(&pt->wctx, backend)) {
        wctx_free(&pt->wctx);
        gf_close(&pt->gguf);
        return false;
    }

    gf_close(&pt->gguf);

    return true;
}

void pipeline_tts_free(PipelineTTS * pt) {
    wctx_free(&pt->wctx);
    *pt = {};
}

// Full LLM forward in a single graph. Composes the custom embed, the 28L
// Qwen3 stack, and the audio_heads reshape. attention_mask is an optional
// [S, S] int 0/1 buffer (1 = attended, 0 = blocked). NULL means
// bidirectional (no padding).
std::vector<float> pipeline_tts_llm_forward(PipelineTTS *   pt,
                                            const int32_t * input_ids,
                                            const int32_t * audio_mask,
                                            const int32_t * attention_mask,
                                            int             K,
                                            int             S,
                                            const char *    dump_hidden_dir,
                                            const char *    dump_hidden_name) {
    if (K <= 0 || S <= 0) {
        return {};
    }
    if (K > pt->lm.num_audio_codebook) {
        fprintf(stderr, "[LM-Forward] FATAL: K=%d exceeds num_audio_codebook=%d\n", K, pt->lm.num_audio_codebook);
        return {};
    }

    const Qwen3Config & cfg = pt->lm.cfg;
    const int           V   = pt->lm.audio_vocab_size;

    // CPU pre-compute, identical to the embed_test pre-compute.
    std::vector<int32_t> shifted((size_t) K * (size_t) S);
    std::vector<int32_t> text_ids_buf(S);
    std::vector<float>   mask_f(S), inv_mask_f(S);
    for (int s = 0; s < S; s++) {
        int m           = (audio_mask[s] != 0) ? 1 : 0;
        mask_f[s]       = (float) m;
        inv_mask_f[s]   = (float) (1 - m);
        text_ids_buf[s] = input_ids[0 * S + s];
        for (int k = 0; k < K; k++) {
            shifted[(size_t) k * (size_t) S + s] = input_ids[(size_t) k * (size_t) S + s] * m + k * V;
        }
    }

    // Convert int 0/1 attention mask to F16 additive bias matching the Python
    // reference. OmniVoice passes a boolean attention_mask to transformers,
    // which promotes True/False to 1.0/0.0 floats and adds it to the attention
    // scores : allowed positions get a +1.0 boost, blocked positions stay at
    // 0.0. This is not a hard mask : every position still contributes to the
    // softmax, the model was trained against this exact bias semantics.
    std::vector<uint16_t> attn_f16;
    if (attention_mask) {
        attn_f16.resize((size_t) S * (size_t) S);
        for (int sq = 0; sq < S; sq++) {
            for (int skv = 0; skv < S; skv++) {
                float v = (attention_mask[(size_t) sq * (size_t) S + (size_t) skv] != 0) ? 1.0f : 0.0f;
                attn_f16[(size_t) sq * (size_t) S + (size_t) skv] = ggml_fp32_to_fp16(v);
            }
        }
    }

    // Node budget : custom embed ~30, 28L stack ~850, audio_heads ~5.
    // 8192 leaves room for longer sequences and future fusions.
    const int    n_max_nodes    = 8192;
    const size_t graph_ctx_size = ggml_tensor_overhead() * n_max_nodes + ggml_graph_overhead_custom(n_max_nodes, false);

    struct ggml_init_params gp   = { graph_ctx_size, NULL, /*no_alloc=*/true };
    struct ggml_context *   gctx = ggml_init(gp);
    if (!gctx) {
        return {};
    }

    // Custom embed inputs.
    struct ggml_tensor * t_text_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_text_ids, "text_ids");
    ggml_set_input(t_text_ids);

    struct ggml_tensor * t_shifted = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, K);
    ggml_set_name(t_shifted, "shifted_ids");
    ggml_set_input(t_shifted);

    struct ggml_tensor * t_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, S);
    ggml_set_name(t_mask, "mask");
    ggml_set_input(t_mask);

    struct ggml_tensor * t_inv_mask = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, 1, S);
    ggml_set_name(t_inv_mask, "inv_mask");
    ggml_set_input(t_inv_mask);

    // Stack input : positions 0..S-1.
    struct ggml_tensor * t_positions = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
    ggml_set_name(t_positions, "positions");
    ggml_set_input(t_positions);

    // Optional attention mask tensor.
    struct ggml_tensor * t_attn = NULL;
    if (attention_mask) {
        t_attn = ggml_new_tensor_2d(gctx, GGML_TYPE_F16, S, S);
        ggml_set_name(t_attn, "attn_mask");
        ggml_set_input(t_attn);
    }

    // Custom embed graph.
    struct ggml_tensor * text_embeds  = ggml_get_rows(gctx, pt->lm.embed_tokens, t_text_ids);
    struct ggml_tensor * audio_embeds = NULL;
    for (int k = 0; k < K; k++) {
        struct ggml_tensor * idx_k = ggml_view_1d(gctx, t_shifted, S, (size_t) k * (size_t) S * sizeof(int32_t));
        struct ggml_tensor * emb_k = ggml_get_rows(gctx, pt->lm.audio_embeddings, idx_k);
        audio_embeds               = (k == 0) ? emb_k : ggml_add(gctx, audio_embeds, emb_k);
    }
    struct ggml_tensor * text_branch   = ggml_mul(gctx, text_embeds, t_inv_mask);
    struct ggml_tensor * audio_branch  = ggml_mul(gctx, audio_embeds, t_mask);
    struct ggml_tensor * inputs_embeds = ggml_add(gctx, text_branch, audio_branch);

    // 28L Qwen3 stack + final RMSNorm. Mask is forwarded through (NULL -> bidir).
    // When dumping is active we also expose the input embedding (pre layer 0)
    // and a few mid stack hidden states so a Python reference can bisect the
    // origin of any drift layer by layer.
    std::vector<int>                  dump_layer_indices;
    std::vector<struct ggml_tensor *> dump_intermediates;
    if (dump_hidden_dir && dump_hidden_name) {
        dump_layer_indices = { 0, 6, 13, 20 };
        ggml_set_name(inputs_embeds, "lm_inputs_embeds");
        ggml_set_output(inputs_embeds);
    }
    struct ggml_tensor * hidden = qwen3_build_layers(
        gctx, cfg, pt->lm.layers, pt->lm.final_norm, inputs_embeds, t_positions, t_attn, S, pt->use_flash_attn,
        pt->clamp_fp16, dump_hidden_dir && dump_hidden_name ? &dump_layer_indices : nullptr,
        dump_hidden_dir && dump_hidden_name ? &dump_intermediates : nullptr);
    if (dump_hidden_dir && dump_hidden_name) {
        for (struct ggml_tensor * t : dump_intermediates) {
            ggml_set_output(t);
        }
        ggml_set_name(hidden, "lm_last_hidden");
        ggml_set_output(hidden);
    }

    // audio_heads readout + reshape to (V, K, S).
    struct ggml_tensor * logits_flat = ggml_mul_mat(gctx, pt->lm.audio_heads, hidden);
    struct ggml_tensor * logits      = ggml_reshape_3d(gctx, logits_flat, V, K, S);
    ggml_set_name(logits, "audio_logits");
    ggml_set_output(logits);

    struct ggml_cgraph * graph = ggml_new_graph_custom(gctx, n_max_nodes, false);
    ggml_build_forward_expand(graph, logits);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(pt->backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        fprintf(stderr, "[LM-Forward] FATAL: gallocr_alloc_graph failed (K=%d S=%d)\n", K, S);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return {};
    }

    ggml_backend_tensor_set(t_text_ids, text_ids_buf.data(), 0, (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_shifted, shifted.data(), 0, (size_t) K * (size_t) S * sizeof(int32_t));
    ggml_backend_tensor_set(t_mask, mask_f.data(), 0, (size_t) S * sizeof(float));
    ggml_backend_tensor_set(t_inv_mask, inv_mask_f.data(), 0, (size_t) S * sizeof(float));

    std::vector<int32_t> pos_data(S);
    for (int i = 0; i < S; i++) {
        pos_data[i] = i;
    }
    ggml_backend_tensor_set(t_positions, pos_data.data(), 0, (size_t) S * sizeof(int32_t));

    if (t_attn) {
        ggml_backend_tensor_set(t_attn, attn_f16.data(), 0, (size_t) S * (size_t) S * sizeof(uint16_t));
    }

    enum ggml_status st = ggml_backend_graph_compute(pt->backend, graph);
    if (st != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[LM-Forward] FATAL: graph_compute status=%d\n", (int) st);
        ggml_gallocr_free(alloc);
        ggml_free(gctx);
        return {};
    }

    const size_t       n = ggml_nelements(logits);
    std::vector<float> out(n);
    ggml_backend_tensor_get(logits, out.data(), 0, n * sizeof(float));

    if (dump_hidden_dir && dump_hidden_name) {
        DebugDumper dbg;
        debug_init(&dbg, dump_hidden_dir);

        auto dump_tensor_2d = [&](struct ggml_tensor * t, const std::string & full_name) {
            const int          dim0  = (int) t->ne[0];
            const int          dim1  = (int) t->ne[1];
            const size_t       numel = (size_t) dim0 * (size_t) dim1;
            std::vector<float> buf(numel);
            ggml_backend_tensor_get(t, buf.data(), 0, numel * sizeof(float));
            // GGML layout : fast axis is dim0, slow axis is dim1. Numpy reads
            // back as [dim1, dim0] row-major, identical to hidden_states[b]
            // and inputs_embeds[b] from the Python reference.
            debug_dump_2d(&dbg, full_name.c_str(), buf.data(), dim1, dim0);
        };

        // Pre layer 0 embedding.
        dump_tensor_2d(inputs_embeds, std::string(dump_hidden_name) + "-embed");

        // Mid stack hidden states, in the order set by dump_layer_indices.
        for (size_t i = 0; i < dump_intermediates.size(); i++) {
            char suffix[32];
            snprintf(suffix, sizeof(suffix), "-l%d", dump_layer_indices[i]);
            dump_tensor_2d(dump_intermediates[i], std::string(dump_hidden_name) + suffix);
        }

        // Final hidden, post output norm, pre lm_head.
        dump_tensor_2d(hidden, dump_hidden_name);
    }

    ggml_gallocr_free(alloc);
    ggml_free(gctx);
    return out;
}

// Batched LLM forward : B' independent forwards stacked. Each batch row may
// carry its own attention_mask, useful for the cond + uncond CFG batching
// where the uncond row uses a diagonal padding past its effective length.
std::vector<float> pipeline_tts_llm_forward_batched(PipelineTTS *   pt,
                                                    const int32_t * input_ids,
                                                    const int32_t * audio_mask,
                                                    const int32_t * attention_mask,
                                                    int             B_prime,
                                                    int             K,
                                                    int             S,
                                                    const char *    dump_hidden_dir) {
    if (B_prime <= 0 || K <= 0 || S <= 0) {
        return {};
    }
    const int          V        = pt->lm.audio_vocab_size;
    const size_t       per_item = (size_t) V * (size_t) K * (size_t) S;
    std::vector<float> out((size_t) B_prime * per_item);

    for (int b = 0; b < B_prime; b++) {
        const int32_t * ids_b  = input_ids + (size_t) b * (size_t) K * (size_t) S;
        const int32_t * mask_b = audio_mask + (size_t) b * (size_t) S;
        const int32_t * attn_b = attention_mask ? attention_mask + (size_t) b * (size_t) S * (size_t) S : NULL;

        // Cond is row 0, uncond is row 1 (CFG batching convention). Map to
        // human readable names so the Python reference can be paired easily.
        const char * hidden_name = nullptr;
        char         hidden_buf[64];
        if (dump_hidden_dir) {
            if (b == 0) {
                hidden_name = "lm-hidden-step0-cond";
            } else if (b == 1) {
                hidden_name = "lm-hidden-step0-uncond";
            } else {
                snprintf(hidden_buf, sizeof(hidden_buf), "lm-hidden-step0-b%d", b);
                hidden_name = hidden_buf;
            }
        }

        std::vector<float> logits_b =
            pipeline_tts_llm_forward(pt, ids_b, mask_b, attn_b, K, S, dump_hidden_dir, hidden_name);
        if (logits_b.size() != per_item) {
            fprintf(stderr, "[LM-Forward] FATAL: batched item %d returned %zu f32 (expected %zu)\n", b, logits_b.size(),
                    per_item);
            return {};
        }
        std::copy(logits_b.begin(), logits_b.end(), out.begin() + (size_t) b * per_item);
    }
    return out;
}

// Public TTS entry. Tokenize text, build prompt + CFG batch via prompt_tts_build,
// run the MaskGIT iterative decoder, return audio_tokens [K, T] flat.
std::vector<int32_t> pipeline_tts_generate(PipelineTTS *         pt,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir) {
    if (T <= 0) {
        fprintf(stderr, "[TTS] FATAL: T=%d must be positive\n", T);
        return {};
    }

    PromptTTS prompt = {};
    if (!prompt_tts_build(&prompt, tok, &pt->lm, text, lang, instruct, T, denoise, ref_text, ref_audio_tokens, ref_T)) {
        return {};
    }

    // Dump cond and uncond input_ids row k=0 for prompt diagnostic. Style and
    // text tokens are duplicated across all K codebooks so k=0 is enough.
    {
        DebugDumper dbg;
        debug_init(&dbg, dump_dir);
        int             ids_shape[1] = { prompt.c_len };
        const int32_t * cond_row     = prompt.input_ids.data();
        const int32_t * uncond_row   = prompt.input_ids.data() + (size_t) prompt.K * (size_t) prompt.c_len;
        debug_dump_i32_as_f32(&dbg, "prompt-cond-ids", cond_row, ids_shape, 1);
        debug_dump_i32_as_f32(&dbg, "prompt-uncond-ids", uncond_row, ids_shape, 1);
    }

    fprintf(stderr, "[TTS] Prompt: B'=%d K=%d S=%d c_len=%d u_len=%d\n", prompt.B_prime, prompt.K, prompt.S_max,
            prompt.c_len, prompt.u_len);

    return maskgit_generate(pt, &prompt, mg_cfg, T, dump_dir);
}

// Full TTS synthesis : tokens via pipeline_tts_generate, waveform via
// pipeline_codec_decode. Refuses to decode partially decoded outputs.
std::vector<float> pipeline_tts_synthesize(PipelineTTS *         pt,
                                           PipelineCodec *       pc,
                                           const BPETokenizer *  tok,
                                           const std::string &   text,
                                           const std::string &   lang,
                                           const std::string &   instruct,
                                           int                   T,
                                           bool                  denoise,
                                           const MaskgitConfig & mg_cfg,
                                           const std::string &   ref_text,
                                           const int32_t *       ref_audio_tokens,
                                           int                   ref_T,
                                           const char *          dump_dir) {
    std::vector<int32_t> tokens = pipeline_tts_generate(pt, tok, text, lang, instruct, T, denoise, mg_cfg, ref_text,
                                                        ref_audio_tokens, ref_T, dump_dir);
    if (tokens.empty()) {
        return {};
    }

    const int K       = pt->lm.num_audio_codebook;
    const int mask_id = pt->lm.audio_mask_id;
    if ((int) tokens.size() != K * T) {
        fprintf(stderr, "[TTS] FATAL: token vector size %zu does not match K*T=%d*%d\n", tokens.size(), K, T);
        return {};
    }
    int n_residual_mask = 0;
    for (int32_t v : tokens) {
        if (v == mask_id) {
            n_residual_mask++;
        }
    }
    if (n_residual_mask) {
        fprintf(stderr, "[TTS] FATAL: %d residual mask tokens left after MaskGIT, refusing to decode\n",
                n_residual_mask);
        return {};
    }

    DebugDumper dbg;
    debug_init(&dbg, dump_dir);
    int tokens_shape[2] = { K, T };
    debug_dump_i32_as_f32(&dbg, "mg-tokens", tokens.data(), tokens_shape, 2);

    fprintf(stderr, "[TTS] Decode: K=%d T=%d expected_samples=%d\n", K, T, T * pc->hop_length);
    std::vector<float> audio = pipeline_codec_decode(pc, tokens.data(), K, T);

    if (!audio.empty()) {
        debug_dump_1d(&dbg, "output-audio", audio.data(), (int) audio.size());
    }
    return audio;
}
