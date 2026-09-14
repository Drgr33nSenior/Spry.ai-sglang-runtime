// SPDX-License-Identifier: MIT
// Embedded by CMake from source hashes, compiler identity and selected target.
#ifndef SPRY_UQ_BUILD_ID
#error "Build through CMake to embed the runtime source identity"
#endif
#ifndef SPRY_UQ_BUILD_TARGET
#error "A build target identity is required"
#endif

extern "C" const char* spry_uq_build_identity() { return SPRY_UQ_BUILD_ID; }
extern "C" const char* spry_uq_build_target() { return SPRY_UQ_BUILD_TARGET; }
