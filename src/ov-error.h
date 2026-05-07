#pragma once
// ov-error.h: internal helper backing the public ov_last_error() entry.
//
// Not part of the public ABI. Translation units that emit user-facing
// errors include this header to record a diagnostic on the calling thread
// before they return a negative ov_status (or NULL). The actual storage
// and the public ov_last_error() reader live in omnivoice.cpp.
//
// Storage is thread_local so concurrent ov_synthesize calls on different
// threads never race on each other's messages. The setter is variadic with
// printf semantics ; messages longer than the internal buffer are
// truncated, never split. Passing NULL as fmt clears the slot.

#include <cstdarg>

void ov_set_error(const char * fmt, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    ;

void ov_set_error_v(const char * fmt, va_list ap);
