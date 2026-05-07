// omnivoice.cpp: public ABI implementation.
//
// Every entry declared in omnivoice.h lives here under one extern "C" block
// so the symbols carry C linkage and are linkable from C, Rust, Go, Python
// ctypes and any other binding generator. The struct ov_context opaque
// handle owns one BackendPair, one PipelineTTS, one PipelineCodec
// (optional), one BPETokenizer and one VoiceDesign instance. ov_init walks
// the load chain in dependency order and unwinds whatever it already
// allocated when any step fails. ov_free mirrors that order in reverse.

#include "omnivoice.h"

#include "backend.h"
#include "bpe.h"
#include "ov-error.h"
#include "pipeline-codec.h"
#include "pipeline-tts.h"
#include "version.h"
#include "voice-design.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Internal definition of the opaque handle. C++ types are fine here
// because nothing in this struct ever crosses the public ABI boundary :
// callers only ever see `struct ov_context *`.
struct ov_context {
    BackendPair   bp;
    PipelineTTS   pt;
    PipelineCodec pc;
    BPETokenizer  tok;
    VoiceDesign   vd;
    bool          codec_loaded;
};

// Thread-local backing store for ov_last_error(). std::string sized once
// per thread, grows on demand, never freed across calls : the std runtime
// reclaims it on thread exit. An empty string means "no error recorded on
// this thread yet", which ov_last_error() exposes as "".
static thread_local std::string g_last_error;

void ov_set_error_v(const char * fmt, va_list ap) {
    if (!fmt) {
        g_last_error.clear();
        return;
    }
    // Two-pass vsnprintf : first call sizes the buffer, second writes the
    // message. va_copy keeps the original ap valid for the second pass.
    va_list ap2;
    va_copy(ap2, ap);
    int needed = std::vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (needed < 0) {
        g_last_error = "ov_set_error : vsnprintf failed";
        return;
    }
    g_last_error.resize(static_cast<size_t>(needed));
    std::vsnprintf(g_last_error.data(), static_cast<size_t>(needed) + 1, fmt, ap);
}

void ov_set_error(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    ov_set_error_v(fmt, ap);
    va_end(ap);
}

extern "C" {

const char * ov_version(void) {
    // Built once at first call. The C++11 magic static guarantees the
    // initialiser runs exactly once across threads.
    static const std::string s = [] {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%d.%d.%d (%s)", OV_VERSION_MAJOR, OV_VERSION_MINOR, OV_VERSION_PATCH,
                      OMNIVOICE_VERSION);
        return std::string(buf);
    }();
    return s.c_str();
}

const char * ov_last_error(void) {
    // c_str() on an empty std::string is guaranteed to point to a NUL
    // byte by C++11, so callers never have to NULL-check the result.
    return g_last_error.c_str();
}

void ov_audio_free(struct ov_audio * a) {
    if (!a) {
        return;
    }
    if (a->samples) {
        std::free(a->samples);
    }
    a->samples     = nullptr;
    a->n_samples   = 0;
    a->sample_rate = 0;
    a->channels    = 0;
}

void ov_init_default_params(struct ov_init_params * p) {
    p->model_path = nullptr;
    p->codec_path = nullptr;
    p->use_fa     = true;
    p->clamp_fp16 = false;
}

void ov_tts_default_params(struct ov_tts_params * p) {
    p->text                    = nullptr;
    p->lang                    = nullptr;
    p->instruct                = nullptr;
    p->T_override              = 0;
    p->chunk_duration_sec      = 15.0f;
    p->chunk_threshold_sec     = 30.0f;
    p->denoise                 = true;
    p->preprocess_prompt       = true;
    p->mg_num_step             = 32;
    p->mg_guidance_scale       = 2.0f;
    p->mg_t_shift              = 0.1f;
    p->mg_layer_penalty_factor = 5.0f;
    p->mg_position_temperature = 5.0f;
    p->mg_class_temperature    = 0.0f;
    p->mg_seed                 = 42;
    p->ref_audio_tokens        = nullptr;
    p->ref_T                   = 0;
    p->ref_audio_24k           = nullptr;
    p->ref_n_samples           = 0;
    p->ref_text                = nullptr;
    p->dump_dir                = nullptr;
    p->cancel                  = nullptr;
    p->cancel_user_data        = nullptr;
}

struct ov_context * ov_init(const struct ov_init_params * params) {
    if (!params || !params->model_path) {
        ov_set_error("ov_init : params or model_path is NULL");
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_init requires a model_path\n");
        return nullptr;
    }

    std::fprintf(stderr, "[OmniVoice] omnivoice.cpp %s\n", ov_version());

    // new ov_context() value-initialises every field : POD aggregates
    // (BackendPair, PipelineTTS, PipelineCodec) are zero-init, std
    // containers in BPETokenizer construct empty, codec_loaded falls to
    // false. Only VoiceDesign needs explicit population below.
    ov_context * ov = new ov_context();

    voice_design_init(&ov->vd);

    // Backend init is shared (refcounted) across modules in the same
    // binary, so ov_init / ov_free pairs balance the refcount cleanly.
    ov->bp = backend_init("LM");
    if (!ov->bp.backend) {
        ov_set_error("ov_init : backend_init failed (no GGML backend available)");
        delete ov;
        return nullptr;
    }

    if (!pipeline_tts_load(&ov->pt, params->model_path, ov->bp, params->use_fa, params->clamp_fp16)) {
        ov_set_error("ov_init : pipeline_tts_load failed for '%s'", params->model_path);
        backend_release(ov->bp.backend, ov->bp.cpu_backend);
        delete ov;
        return nullptr;
    }

    // BPE tokenizer payload lives inside the same LM GGUF as the weights.
    // Load the base vocab + the OmniVoice-specific special tokens in one
    // shot.
    if (!load_bpe_from_gguf(&ov->tok, params->model_path) ||
        !bpe_load_omnivoice_specials(&ov->tok, params->model_path)) {
        ov_set_error("ov_init : BPE / OmniVoice specials load failed for '%s'", params->model_path);
        pipeline_tts_free(&ov->pt);
        backend_release(ov->bp.backend, ov->bp.cpu_backend);
        delete ov;
        return nullptr;
    }

    if (params->codec_path) {
        if (!pipeline_codec_load(&ov->pc, params->codec_path, ov->bp)) {
            ov_set_error("ov_init : pipeline_codec_load failed for '%s'", params->codec_path);
            pipeline_tts_free(&ov->pt);
            backend_release(ov->bp.backend, ov->bp.cpu_backend);
            delete ov;
            return nullptr;
        }
        ov->codec_loaded = true;
    }

    return ov;
}

void ov_free(struct ov_context * ov) {
    if (!ov) {
        return;
    }
    if (ov->codec_loaded) {
        pipeline_codec_free(&ov->pc);
    }
    pipeline_tts_free(&ov->pt);
    backend_release(ov->bp.backend, ov->bp.cpu_backend);
    delete ov;
}

enum ov_status ov_synthesize(struct ov_context * ov, const struct ov_tts_params * params, struct ov_audio * out) {
    if (!ov || !params || !out) {
        ov_set_error("ov_synthesize : ov / params / out is NULL");
        if (out) {
            ov_audio_free(out);
        }
        return OV_STATUS_INVALID_PARAMS;
    }
    if (!ov->codec_loaded) {
        ov_set_error("ov_synthesize : codec not loaded (pass codec_path to ov_init)");
        ov_audio_free(out);
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_synthesize requires a codec-loaded handle\n");
        return OV_STATUS_INVALID_PARAMS;
    }
    return pipeline_tts_synthesize(&ov->pt, &ov->pc, &ov->tok, &ov->vd, params, out);
}

int ov_duration_sec_to_tokens(const struct ov_context * ov, float duration_sec) {
    if (!ov || !ov->codec_loaded) {
        ov_set_error("ov_duration_sec_to_tokens : codec not loaded");
        std::fprintf(stderr, "[OmniVoice] ERROR: ov_duration_sec_to_tokens requires a codec-loaded handle\n");
        return 1;
    }
    return pipeline_tts_duration_sec_to_tokens(&ov->pc, duration_sec);
}

}  // extern "C"
