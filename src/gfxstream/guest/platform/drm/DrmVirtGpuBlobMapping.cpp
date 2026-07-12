/*
 * Copyright 2022 Google
 * SPDX-License-Identifier: MIT
 */

#include <sys/mman.h>

#include <cstdint>

#include "DrmVirtGpu.h"
#include "drm-uapi/virtgpu_drm.h"

// Defined in DrmVirtGpuBlob.cpp. Returns a freed GFXSTREAM_MAP_LOW slot to the
// low-VA arena for reuse; a no-op for high (non-arena) mappings and when the
// feature is off.
void gfxstreamLowVaRelease(void* addr, uint64_t size);

DrmVirtGpuResourceMapping::DrmVirtGpuResourceMapping(VirtGpuResourcePtr blob, uint8_t* ptr,
                                                         uint64_t size)
    : mBlob(blob), mPtr(ptr), mSize(size) {}

DrmVirtGpuResourceMapping::~DrmVirtGpuResourceMapping(void) {
    munmap(mPtr, mSize);
    gfxstreamLowVaRelease(mPtr, mSize);
}

uint8_t* DrmVirtGpuResourceMapping::asRawPtr(void) { return mPtr; }
