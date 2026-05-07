/* tests/abi-c.c: link-only ABI smoke test for omnivoice.h.
 *
 * Compiled in pure C99 with -Wall -Werror -pedantic. The purpose of this
 * test is NOT to run a full synthesis (no GGUF loaded, no model required) ;
 * it is to guarantee at every build that :
 *
 *   1. omnivoice.h parses with a C compiler (no <cstdio>, no std::*, no
 *      C++-only forward declarations leak in).
 *   2. Every public ov_* symbol has C linkage and links from a C
 *      translation unit.
 *   3. The structs are POD and zero-initialisable with `{0}` from C.
 *
 * If this test stops compiling or stops linking, the public ABI has
 * regressed and the build breaks before anything else.
 */

#include "omnivoice.h"

#include <stdio.h>
#include <stdlib.h>

static bool stub_cancel(void * ud) {
    (void) ud;
    return false;
}

int main(void) {
    /* Static version string, always reachable. */
    const char * version = ov_version();
    printf("omnivoice ABI probe : %s\n", version);

    /* Default-initialise the public structs from C. */
    struct ov_init_params iparams;
    ov_init_default_params(&iparams);

    struct ov_tts_params params;
    ov_tts_default_params(&params);

    /* Sanity-check a few default values. */
    if (params.mg_num_step != 32 || params.chunk_duration_sec <= 0.0f) {
        fprintf(stderr, "ABI probe : default values do not match\n");
        return 1;
    }

    /* Touch every reference-pointer field, every callback typedef and
     * every output struct field so the compiler validates the layout
     * end-to-end without ever needing a model. */
    params.cancel           = stub_cancel;
    params.cancel_user_data = NULL;

    struct ov_audio audio = { 0 };
    ov_audio_free(&audio);

    /* Call every entry through its early-return path. ov_init returns
     * NULL on missing model_path, ov_synthesize / ov_duration_sec_to_tokens
     * fail on NULL handle, ov_free is safe on NULL. None of these load a
     * model, but the linker must resolve every name to satisfy the call. */
    struct ov_context * dummy = ov_init(NULL);
    if (dummy != NULL) {
        fprintf(stderr, "ABI probe : ov_init(NULL) was supposed to return NULL\n");
        ov_free(dummy);
        return 2;
    }

    /* ov_init(NULL) just failed -> ov_last_error() must point to a
     * non-empty thread-local string. Pointer is always valid (c_str on
     * an empty std::string still gives a NUL byte), so we only need to
     * check the first byte to confirm an error was actually recorded. */
    const char * err = ov_last_error();
    if (err == NULL || err[0] == '\0') {
        fprintf(stderr, "ABI probe : ov_last_error() empty after a known failure\n");
        return 5;
    }
    printf("omnivoice ABI probe : ov_last_error reads '%s'\n", err);

    enum ov_status rc = ov_synthesize(NULL, &params, &audio);
    if (rc != OV_STATUS_INVALID_PARAMS) {
        fprintf(stderr, "ABI probe : ov_synthesize(NULL) returned %d, expected %d\n", (int) rc,
                (int) OV_STATUS_INVALID_PARAMS);
        return 3;
    }

    int frames = ov_duration_sec_to_tokens(NULL, 1.0f);
    if (frames < 1) {
        fprintf(stderr, "ABI probe : ov_duration_sec_to_tokens returned %d, expected >= 1\n", frames);
        return 4;
    }

    ov_free(NULL);
    ov_audio_free(&audio);

    return 0;
}
