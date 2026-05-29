// omnivoice diagnostic — exposes eliza_diagnostic_self_check() and
// eliza_diagnostic_set_logger() so libomnivoice.so can be dlopen-smoke-
// tested under qemu-riscv64-static without loading any model.
//
// Contract: include/eliza_diagnostic.h. Helpers: diagnostic/eliza_diag_helpers.h.

#include "../../include/eliza_diagnostic.h"
#include "../../diagnostic/eliza_diag_helpers.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "audio-resample.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kJsonCap = 4096;
thread_local char g_json[kJsonCap];

std::atomic<eliza_diag_log_fn> g_logger{nullptr};

void emit_log(const char* line) {
    eliza_diag_log_fn fn = g_logger.load(std::memory_order_acquire);
    if (fn) fn(line);
}

struct Check {
    const char* name;
    bool ok;
    const char* detail;
};

int render_check(char* buf, std::size_t cap, std::size_t pos, bool first, const Check& c) {
    return std::snprintf(buf + pos, cap > pos ? cap - pos : 0,
        "%s{\"name\":\"%s\",\"ok\":%s,\"detail\":\"%s\"}",
        first ? "" : ",",
        c.name, c.ok ? "true" : "false", c.detail);
}

bool check_backend_init(const char*& detail) {
    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { detail = "ggml_backend_cpu_init returned NULL"; return false; }
    ggml_backend_free(be);
    detail = "cpu backend init+free ok";
    return true;
}

bool check_resample_roundtrip(const char*& detail) {
    // 48000 -> 16000 is the canonical mic-input downsample the
    // pipeline-codec / pipeline-tts paths perform. Validate the
    // public audio_resample() entry point produces the expected number
    // of output samples for a known-length input. With 3 samples in at
    // 48 kHz the kernel-padded output at 16 kHz is 1 sample (3 * 16000
    // / 48000 = 1 after truncation).
    const float in[3] = { 0.0f, 0.5f, -0.5f };
    int n_out = 0;
    float* out = audio_resample(in, 3, 48000, 16000, 1, &n_out);
    if (!out) { detail = "audio_resample returned NULL"; return false; }
    const bool ok_len = (n_out == 1);
    std::free(out);
    if (!ok_len) { detail = "unexpected n_out for 3@48k->16k"; return false; }
    detail = "48000->16000 mono roundtrip ok";
    return true;
}

bool check_resample_kernel_constants(const char*& detail) {
    // The torchaudio-compatible resampler exposes its filter-window length
    // and rolloff as compile-time constants. Lock them down so a refactor
    // that silently changes either tripwires the diagnostic.
    if (AUDIO_RESAMPLE_LPFW != 6) { detail = "AUDIO_RESAMPLE_LPFW changed"; return false; }
    constexpr double rolloff = AUDIO_RESAMPLE_ROLLOFF;
    if (rolloff < 0.98 || rolloff > 1.0) { detail = "AUDIO_RESAMPLE_ROLLOFF out of range"; return false; }
    detail = "resample kernel constants stable";
    return true;
}

bool check_backend_count(const char*& detail) {
    const std::size_t n = ggml_backend_reg_count();
    if (n == 0) { detail = "no ggml backends registered"; return false; }
    static char buf[64];
    std::snprintf(buf, sizeof buf, "%zu backends registered", n);
    detail = buf;
    return true;
}

}  // namespace

extern "C" __attribute__((visibility("default")))
void eliza_diagnostic_set_logger(eliza_diag_log_fn fn) {
    g_logger.store(fn, std::memory_order_release);
}

extern "C" __attribute__((visibility("default")))
const char* eliza_diagnostic_self_check(void) {
    struct EzDiagCpu cpu;
    eliza_diag_probe_cpu(&cpu);

    Check checks[4];
    int n = 0;
    bool all_ok = true;

    {
        const char* d = "";
        bool ok = check_backend_init(d);
        checks[n++] = { "ggml_backend_cpu_init_free", ok, d };
        if (!ok) all_ok = false;
        emit_log(ok ? "[omnivoice_diag] ggml backend ok" : "[omnivoice_diag] ggml backend FAILED");
    }
    {
        const char* d = "";
        bool ok = check_backend_count(d);
        checks[n++] = { "ggml_backend_reg_nonempty", ok, d };
        if (!ok) all_ok = false;
    }
    {
        const char* d = "";
        bool ok = check_resample_kernel_constants(d);
        checks[n++] = { "audio_resample_constants", ok, d };
        if (!ok) all_ok = false;
    }
    {
        const char* d = "";
        bool ok = check_resample_roundtrip(d);
        checks[n++] = { "audio_resample_48k_to_16k", ok, d };
        if (!ok) all_ok = false;
        emit_log(ok ? "[omnivoice_diag] resample ok" : "[omnivoice_diag] resample FAILED");
    }

    std::size_t pos = 0;
    int w = std::snprintf(g_json, kJsonCap, "{");
    pos += (w > 0 ? static_cast<std::size_t>(w) : 0);

    char header[1024];
    eliza_diag_render_header(header, sizeof header, "omnivoice", "", &cpu);
    w = std::snprintf(g_json + pos, kJsonCap > pos ? kJsonCap - pos : 0, "%s", header);
    pos += (w > 0 ? static_cast<std::size_t>(w) : 0);

    w = std::snprintf(g_json + pos, kJsonCap > pos ? kJsonCap - pos : 0, ",\"checks\":[");
    pos += (w > 0 ? static_cast<std::size_t>(w) : 0);

    for (int i = 0; i < n; i++) {
        w = render_check(g_json, kJsonCap, pos, i == 0, checks[i]);
        pos += (w > 0 ? static_cast<std::size_t>(w) : 0);
    }

    w = std::snprintf(g_json + pos, kJsonCap > pos ? kJsonCap - pos : 0,
                      "],\"ok\":%s}", all_ok ? "true" : "false");
    pos += (w > 0 ? static_cast<std::size_t>(w) : 0);

    if (pos >= kJsonCap) g_json[kJsonCap - 1] = '\0';
    return g_json;
}
