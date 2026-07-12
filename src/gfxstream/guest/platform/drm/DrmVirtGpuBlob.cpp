/*
 * Copyright 2022 Google
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "DrmVirtGpu.h"
#include "drm-uapi/virtgpu_drm.h"
#include "util/log.h"

// As per the warning in xf86drm.h, callers of drmPrimeFDToHandle are expected to perform
// reference counting on the underlying GEM handle that is returned. With Vulkan, for example,
// it is entirely possible that an FD, which points to the same underlying GEM handle, is both
// exported then imported across Vulkan objects. As the VirtGpuResource is stored as a
// shared_ptr with it's own ref-countng, the ref-counting for the underyling GEM has to be
// internal to this implementation. Otherwise, a GEM handle which is active in another Vulkan object
// in the same process, may be erroneously closed in the desctructor of one of the shared ptrs.
static std::mutex sDrmObjectRefMutex;
static std::unordered_map<uint32_t, int> sDrmObjectRefMap;

// ---------------------------------------------------------------------------
// Low-VA arena for GFXSTREAM_MAP_LOW (32-bit host-visible mappings). Hands out
// page-aligned addresses in [1GiB, ~3.75GiB) so a 32-bit WoW64/FEX caller can
// hold the pointer. Freed slots (returned by the mapping destructor via
// gfxstreamLowVaRelease) are recycled by exact size, so churn does not march
// the bump pointer to the top of the window. Defined here (external linkage)
// so DrmVirtGpuBlobMapping.cpp's destructor can release into the same arena.
// ---------------------------------------------------------------------------
namespace {
constexpr uintptr_t kLowVaBase = 0x40000000ULL;  // 1 GiB (above the program image)
constexpr uintptr_t kLowVaTop = 0xF0000000ULL;   // < 4 GiB with headroom

inline uint64_t lowVaAligned(uint64_t size) { return (size + 0xFFF) & ~uint64_t(0xFFF); }

struct LowVaArena {
    std::mutex mu;
    uintptr_t next = kLowVaBase;
    std::unordered_map<uint64_t, std::vector<uintptr_t>> freeBySize;

    uintptr_t reserve(uint64_t aligned) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = freeBySize.find(aligned);
        if (it != freeBySize.end() && !it->second.empty()) {
            uintptr_t a = it->second.back();
            it->second.pop_back();
            return a;
        }
        if (next + aligned < kLowVaTop) {
            uintptr_t a = next;
            next += aligned;
            return a;
        }
        return 0;  // exhausted
    }
    void release(uintptr_t addr, uint64_t aligned) {
        std::lock_guard<std::mutex> lk(mu);
        freeBySize[aligned].push_back(addr);
    }
};

LowVaArena& lowVaArena() {
    static LowVaArena arena;
    return arena;
}
}  // namespace

bool gfxstreamLowVaEnabled() {
    static const bool enabled = [] {
        const char* e = getenv("GFXSTREAM_MAP_LOW");
        return e && e[0] == '1';
    }();
    return enabled;
}

uintptr_t gfxstreamLowVaReserve(uint64_t size) {
    return lowVaArena().reserve(lowVaAligned(size));
}

void gfxstreamLowVaRelease(void* addr, uint64_t size) {
    if (!gfxstreamLowVaEnabled()) return;
    uintptr_t a = reinterpret_cast<uintptr_t>(addr);
    if (a < kLowVaBase || a >= kLowVaTop) return;  // not one of ours (high fallback)
    lowVaArena().release(a, lowVaAligned(size));
}

DrmVirtGpuResource::DrmVirtGpuResource(int64_t deviceHandle, uint32_t blobHandle,
                                           uint32_t resourceHandle, uint64_t size)
    : mDeviceHandle(deviceHandle),
      mBlobHandle(blobHandle),
      mResourceHandle(resourceHandle),
      mSize(size) {
    const std::lock_guard<std::mutex> lock(sDrmObjectRefMutex);
    auto refMapIt = sDrmObjectRefMap.find(blobHandle);
    if (refMapIt == sDrmObjectRefMap.end()) {
        sDrmObjectRefMap[blobHandle] = 1;
    } else {
        refMapIt->second++;
    }
}

DrmVirtGpuResource::~DrmVirtGpuResource() {
    if (mBlobHandle == INVALID_DESCRIPTOR) {
        return;
    }

    const std::lock_guard<std::mutex> lock(sDrmObjectRefMutex);
    auto refMapIt = sDrmObjectRefMap.find(mBlobHandle);
    if (refMapIt == sDrmObjectRefMap.end()) {
        mesa_logw(
            "DrmVirtGpuResource::~DrmVirtGpuResource() could not find the blobHandle: %d in "
            "internal map",
            mBlobHandle);
        return;
    }

    refMapIt->second--;
    if (refMapIt->second <= 0) {
        sDrmObjectRefMap.erase(refMapIt);

        struct drm_gem_close gem_close {
            .handle = mBlobHandle, .pad = 0,
        };

        int ret = drmIoctl(mDeviceHandle, DRM_IOCTL_GEM_CLOSE, &gem_close);
        if (ret) {
            mesa_loge("DRM_IOCTL_GEM_CLOSE failed with : [%s, blobHandle %u, resourceHandle: %u]",
                      strerror(errno), mBlobHandle, mResourceHandle);
        }
    }
}

void DrmVirtGpuResource::intoRaw() {
    mBlobHandle = INVALID_DESCRIPTOR;
    mResourceHandle = INVALID_DESCRIPTOR;
}

uint32_t DrmVirtGpuResource::getBlobHandle() const { return mBlobHandle; }

uint32_t DrmVirtGpuResource::getResourceHandle() const { return mResourceHandle; }

uint64_t DrmVirtGpuResource::getSize() const { return mSize; }

VirtGpuResourceMappingPtr DrmVirtGpuResource::createMapping() {
    int ret;
    struct drm_virtgpu_map map {
        .handle = mBlobHandle, .pad = 0,
    };

    ret = drmIoctl(mDeviceHandle, DRM_IOCTL_VIRTGPU_MAP, &map);
    if (ret) {
        mesa_loge("DRM_IOCTL_VIRTGPU_MAP failed with %s", strerror(errno));
        return nullptr;
    }

    // DroidVM experiment (GFXSTREAM_MAP_LOW=1): place host-visible mappings in
    // the low 4GB so a 32-bit WoW64/FEX caller can hold the pointer (wine
    // win32u rejects >4GB maps as "does not fit 32-bit pointer"). The low-VA
    // arena (below) recycles freed slots so a long run does not exhaust the low
    // window. Default off -> zero effect on 64-bit apps.
#if defined(MAP_FIXED_NOREPLACE)
    if (gfxstreamLowVaEnabled()) {
        for (int tries = 0; tries < 128; ++tries) {
            uintptr_t slot = gfxstreamLowVaReserve(mSize);
            if (!slot) break;  // arena exhausted
            void* got = mmap(reinterpret_cast<void*>(slot), mSize, PROT_WRITE | PROT_READ,
                             MAP_SHARED | MAP_FIXED_NOREPLACE, mDeviceHandle, map.offset);
            if (got != MAP_FAILED && reinterpret_cast<uintptr_t>(got) == slot) {
                mesa_logw("GFXSTREAM_MAP_LOW: mapped %llu bytes at %p (low)",
                          (unsigned long long)mSize, got);
                return std::make_shared<DrmVirtGpuResourceMapping>(shared_from_this(),
                                                                   static_cast<uint8_t*>(got), mSize);
            }
            // Slot was taken out from under us (FEX/other): drop it (do NOT
            // recycle) and ask the arena for the next one.
            if (got != MAP_FAILED) munmap(got, mSize);
        }
        mesa_logw("GFXSTREAM_MAP_LOW: no free low slot (size=%llu), falling back to high",
                  (unsigned long long)mSize);
    }
#endif

    uint8_t* ptr = static_cast<uint8_t*>(
        mmap(nullptr, mSize, PROT_WRITE | PROT_READ, MAP_SHARED, mDeviceHandle, map.offset));

    if (ptr == MAP_FAILED) {
        mesa_loge("mmap64 failed with (%s)", strerror(errno));
        return nullptr;
    }

    mesa_logw("GFXSTREAM_MAP: mapped %llu bytes at %p", (unsigned long long)mSize, ptr);
    return std::make_shared<DrmVirtGpuResourceMapping>(shared_from_this(), ptr, mSize);
}

int DrmVirtGpuResource::exportBlob(struct VirtGpuExternalHandle& handle) {
    int ret, fd;

    uint32_t flags = DRM_CLOEXEC;
    ret = drmPrimeHandleToFD(mDeviceHandle, mBlobHandle, flags, &fd);
    if (ret) {
        mesa_loge("drmPrimeHandleToFD failed with %s", strerror(errno));
        return ret;
    }

    handle.osHandle = static_cast<int64_t>(fd);
    handle.type = kMemHandleDmabuf;
    return 0;
}

int DrmVirtGpuResource::wait() {
    int ret;
    struct drm_virtgpu_3d_wait wait_3d = {0};

    int retry = 0;
    do {
        if (retry > 0 && (retry % 10 == 0)) {
            mesa_loge("DRM_IOCTL_VIRTGPU_WAIT failed with EBUSY for %d times.", retry);
        }
        wait_3d.handle = mBlobHandle;
        ret = drmIoctl(mDeviceHandle, DRM_IOCTL_VIRTGPU_WAIT, &wait_3d);
        ++retry;
    } while (ret < 0 && errno == EBUSY);

    if (ret < 0) {
        mesa_loge("DRM_IOCTL_VIRTGPU_WAIT failed with %s", strerror(errno));
        return ret;
    }

    return 0;
}

int DrmVirtGpuResource::transferToHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    int ret;
    struct drm_virtgpu_3d_transfer_to_host xfer = {0};

    xfer.box.x = x;
    xfer.box.y = y;
    xfer.box.w = w;
    xfer.box.h = h;
    xfer.box.d = 1;
    xfer.bo_handle = mBlobHandle;

    ret = drmIoctl(mDeviceHandle, DRM_IOCTL_VIRTGPU_TRANSFER_TO_HOST, &xfer);
    if (ret < 0) {
        mesa_loge("DRM_IOCTL_VIRTGPU_TRANSFER_TO_HOST failed with %s", strerror(errno));
        return ret;
    }

    return 0;
}

int DrmVirtGpuResource::transferFromHost(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    int ret;
    struct drm_virtgpu_3d_transfer_from_host xfer = {0};

    xfer.box.x = x;
    xfer.box.y = y;
    xfer.box.w = w;
    xfer.box.h = h;
    xfer.box.d = 1;
    xfer.bo_handle = mBlobHandle;

    ret = drmIoctl(mDeviceHandle, DRM_IOCTL_VIRTGPU_TRANSFER_FROM_HOST, &xfer);
    if (ret < 0) {
        mesa_loge("DRM_IOCTL_VIRTGPU_TRANSFER_FROM_HOST failed with %s", strerror(errno));
        return ret;
    }

    return 0;
}
