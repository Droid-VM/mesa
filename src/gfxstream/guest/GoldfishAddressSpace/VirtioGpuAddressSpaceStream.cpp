/*
 * Copyright 2023 Google
 * SPDX-License-Identifier: MIT
 */

#include "VirtioGpuAddressSpaceStream.h"

#include <unistd.h>

#include <errno.h>

#include "util/log.h"
#include "util/os_misc.h"

static bool GetRingParamsFromCapset(enum VirtGpuCapset capset, const VirtGpuCaps& caps,
                                    uint32_t& ringSize, uint32_t& bufferSize,
                                    uint32_t& blobAlignment) {
    uint64_t guest_page_size = 4096;
    os_get_page_size(&guest_page_size);
    switch (capset) {
        case kCapsetGfxStreamVulkan:
            ringSize = caps.vulkanCapset.ringSize;
            bufferSize = caps.vulkanCapset.bufferSize;
            blobAlignment = caps.vulkanCapset.blobAlignment;
            break;
        case kCapsetGfxStreamGles:
            ringSize = caps.glesCapset.ringSize;
            bufferSize = caps.glesCapset.bufferSize;
            blobAlignment = caps.glesCapset.blobAlignment;
            break;
        case kCapsetGfxStreamComposer:
            ringSize = caps.composerCapset.ringSize;
            bufferSize = caps.composerCapset.bufferSize;
            blobAlignment = caps.composerCapset.blobAlignment;
            break;
        default:
            return false;
    }

    blobAlignment = std::max(blobAlignment, (uint32_t)guest_page_size);
    return true;
}

static address_space_handle_t virtgpu_address_space_open() {
    return (address_space_handle_t)(-EINVAL);
}

static void virtgpu_address_space_close(address_space_handle_t) {
    // Handle opened by VirtioGpuDevice wrapper
}

static bool virtgpu_address_space_ping(address_space_handle_t, struct address_space_ping* info) {
    int ret;
    struct VirtGpuExecBuffer exec = {};
    VirtGpuDevice* instance = VirtGpuDevice::getInstance();
    struct gfxstreamContextPing ping = {};

    ping.hdr.opCode = GFXSTREAM_CONTEXT_PING;
    ping.resourceId = info->resourceId;

    exec.command = static_cast<void*>(&ping);
    exec.command_size = sizeof(ping);

    ret = instance->execBuffer(exec, nullptr);
    if (ret)
        return false;

    return true;
}

AddressSpaceStream* createVirtioGpuAddressSpaceStream(enum VirtGpuCapset capset) {
    VirtGpuResourcePtr pipe, blob;
    VirtGpuResourceMappingPtr pipeMapping, blobMapping;
    struct VirtGpuExecBuffer exec = {};
    struct VirtGpuCreateBlob blobCreate = {};
    struct gfxstreamContextCreate contextCreate = {};

    uint32_t ringSize = 0;
    uint32_t bufferSize = 0;
    uint32_t blobAlignment = 0;

    char* blobAddr, *bufferPtr;
    int ret;

    VirtGpuDevice* instance = VirtGpuDevice::getInstance();
    auto caps = instance->getCaps();

    if (!caps.params[kParamResourceBlob]) {
        mesa_loge("VirtGpuDevice does not support blob-resources!");
        return nullptr;
    }
    if (!caps.params[kParamHostVisible] && !caps.params[kParamCreateGuestHandle]) {
        mesa_loge("VirtGpuDevice must support at least one of: 1) host-visible memory (kParamHostVisible) or 2) host handles created from guest memory (kParamCreateGuestHandle).");
        return nullptr;
    }

    if (!GetRingParamsFromCapset(capset, caps, ringSize, bufferSize, blobAlignment)) {
        mesa_loge("Failed to get ring parameters");
        return nullptr;
    }

    blobCreate.blobId = 0;
    blobCreate.blobMem = kBlobMemHost3d;
    blobCreate.flags = kBlobFlagMappable;
    blobCreate.size = ALIGN_POT(ringSize + bufferSize, blobAlignment);
    blob = instance->createBlob(blobCreate);
    if (!blob)
        return nullptr;

    // Context creation command
    contextCreate.hdr.opCode = GFXSTREAM_CONTEXT_CREATE;
    contextCreate.resourceId = blob->getResourceHandle();

    exec.command = static_cast<void*>(&contextCreate);
    exec.command_size = sizeof(contextCreate);

    ret = instance->execBuffer(exec, blob.get());
    if (ret)
        return nullptr;

    // Wait occurs on global timeline -- should we use context specific one?
    ret = blob->wait();
    if (ret)
        return nullptr;

    blobMapping = blob->createMapping();
    if (!blobMapping)
        return nullptr;

    blobAddr = reinterpret_cast<char*>(blobMapping->asRawPtr());

    bufferPtr = blobAddr + sizeof(struct asg_ring_storage);
    struct asg_context context = asg_context_create(blobAddr, bufferPtr, bufferSize);

    context.ring_config->transfer_mode = 1;
    context.ring_config->host_consumed_pos = 0;
    context.ring_config->guest_write_pos = 0;

    // flush_interval is published by the host when it creates its consumer, and the stream below
    // reads it as its write step. Reading it too early gives zero, and a stream with a zero write
    // step never uses the ring at all -- every allocation takes the temporary-buffer path and
    // leaves through the large-transfer ring, while the write cursor never advances. Nothing in
    // the setup sequence orders the two, so wait for it rather than assume it.
    {
        int waited = 0;
        while (!__atomic_load_n(&context.ring_config->flush_interval, __ATOMIC_ACQUIRE)) {
            if (++waited > 20000) {
                mesa_loge("gfxstream: host never published a flush interval for this stream");
                return nullptr;
            }
            usleep(100);
        }
        if (waited) {
            mesa_logw("gfxstream: waited %d00us for the host to publish the ring config", waited);
        }
    }

    struct address_space_ops ops = {
        .open = virtgpu_address_space_open,
        .close = virtgpu_address_space_close,
        .ping = virtgpu_address_space_ping,
    };

    AddressSpaceStream* res =
        new AddressSpaceStream((address_space_handle_t)(-1), 1, context, 0, 0, ops);

    res->setMapping(blobMapping);
    res->setResourceId(contextCreate.resourceId);
    return res;
}
