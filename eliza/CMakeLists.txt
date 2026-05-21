# omnivoice.cpp/eliza/CMakeLists.txt
#
# Injected into the upstream omnivoice.cpp build via
#   -DCMAKE_PROJECT_omnivoice-ggml_INCLUDE=<path-to-this-file>
# which CMake evaluates immediately after project() runs. We defer the
# actual target_sources() / target_link_libraries() calls until end-of-
# DIRECTORY so the `omnivoice` shared target defined later by the upstream
# CMakeLists.txt already exists.
#
# Result: libomnivoice.so picks up the diagnostic TU + linked
# eliza_diag_helpers static archive without us editing any upstream file.

set(_eliza_diag_dir          "${CMAKE_CURRENT_LIST_DIR}/../../diagnostic")
set(_eliza_diag_include_dir  "${CMAKE_CURRENT_LIST_DIR}/../../include")
set(_eliza_omnivoice_diag_src "${CMAKE_CURRENT_LIST_DIR}/omnivoice_diagnostic.cpp")

if(NOT TARGET eliza_diag_helpers_omnivoice)
    add_library(eliza_diag_helpers_omnivoice STATIC
        "${_eliza_diag_dir}/eliza_diag_helpers.c")
    set_target_properties(eliza_diag_helpers_omnivoice PROPERTIES
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(eliza_diag_helpers_omnivoice PUBLIC
        "${_eliza_diag_dir}"
        "${_eliza_diag_include_dir}")
endif()

# `omnivoice` shared target is added later in the parent CMakeLists.txt
# (gated on OMNIVOICE_SHARED). Defer the attachment so we run after it
# exists. cmake_language(DEFER) lands at end-of-current-directory; the
# parent project's directory IS current at this include time.
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL
    _eliza_attach_omnivoice_diagnostic)

function(_eliza_attach_omnivoice_diagnostic)
    if(NOT TARGET omnivoice)
        message(WARNING
            "[eliza-diag] omnivoice SHARED target not found; "
            "eliza_diagnostic_self_check will NOT be exported. "
            "Configure with -DOMNIVOICE_SHARED=ON.")
        return()
    endif()
    target_sources(omnivoice PRIVATE "${_eliza_omnivoice_diag_src}")
    target_include_directories(omnivoice PRIVATE
        "${_eliza_diag_dir}"
        "${_eliza_diag_include_dir}"
        "${CMAKE_SOURCE_DIR}/src"
        "${CMAKE_SOURCE_DIR}/ggml/include")
    target_link_libraries(omnivoice PRIVATE eliza_diag_helpers_omnivoice)
    message(STATUS "[eliza-diag] attached omnivoice_diagnostic.cpp to omnivoice shared target")
endfunction()
