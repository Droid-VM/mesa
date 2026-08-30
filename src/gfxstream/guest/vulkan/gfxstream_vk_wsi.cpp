/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#include "util/macros.h"

#include "gfxstream_vk_entrypoints.h"
#include "gfxstream_vk_private.h"
#include "wsi_common.h"

/*
 * VK_KHR_display presents by page-flipping the swapchain image's dma-buf on a DRM device, which
 * needs the *primary* node (the render node the rest of the driver uses cannot modeset). Find the
 * guest's virtio-gpu card node the same way DrmVirtGpuDevice picks its render node. Returns -1
 * when there is none or it cannot be opened -- mesa's WSI then reports zero displays, which is
 * exactly what a headless guest should look like.
 */
static int gfxstream_vk_open_display_fd(void) {
#if defined(GFXSTREAM_VK_DISPLAY)
    drmDevicePtr devs[8];
    int count = drmGetDevices2(0, devs, ARRAY_SIZE(devs));
    if (count <= 0) return -1;

    int display_fd = -1;
    for (int i = 0; i < count && display_fd < 0; i++) {
        drmDevicePtr dev = devs[i];
        if (!(dev->available_nodes & (1 << DRM_NODE_PRIMARY))) continue;

        int fd = open(dev->nodes[DRM_NODE_PRIMARY], O_RDWR | O_CLOEXEC);
        if (fd < 0) continue;

        drmVersionPtr version = drmGetVersion(fd);
        if (version && !strcmp(version->name, "virtio_gpu")) {
            display_fd = fd;
        } else {
            close(fd);
        }
        if (version) drmFreeVersion(version);
    }

    drmFreeDevices(devs, count);
    return display_fd;
#else
    return -1;
#endif
}

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
gfxstream_vk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char* pName) {
    VK_FROM_HANDLE(gfxstream_vk_physical_device, pdevice, physicalDevice);
    return vk_instance_get_proc_addr_unchecked(&pdevice->instance->vk, pName);
}

VkResult gfxstream_vk_wsi_init(struct gfxstream_vk_physical_device* physical_device) {
    VkResult result = (VkResult)0;

    physical_device->display_fd = gfxstream_vk_open_display_fd();

    const struct wsi_device_options options = {.sw_device = false};
    result = wsi_device_init(&physical_device->wsi_device,
                             gfxstream_vk_physical_device_to_handle(physical_device),
                             gfxstream_vk_wsi_proc_addr, &physical_device->instance->vk.alloc,
                             physical_device->display_fd, NULL, &options);
    if (result != VK_SUCCESS) {
        if (physical_device->display_fd >= 0) close(physical_device->display_fd);
        physical_device->display_fd = -1;
        return result;
    }

    // Allow guest-side modifier code paths
    physical_device->wsi_device.supports_modifiers = true;
    // Support wsi_image_create_info::scanout
    physical_device->wsi_device.supports_scanout = true;

    physical_device->vk.wsi_device = &physical_device->wsi_device;

    return result;
}

void gfxstream_vk_wsi_finish(struct gfxstream_vk_physical_device* physical_device) {
    physical_device->vk.wsi_device = NULL;
    wsi_device_finish(&physical_device->wsi_device, &physical_device->instance->vk.alloc);
    if (physical_device->display_fd >= 0) {
        close(physical_device->display_fd);
        physical_device->display_fd = -1;
    }
}
