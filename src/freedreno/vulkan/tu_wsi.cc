/*
 * Copyright © 2016 Red Hat
 * SPDX-License-Identifier: MIT
 *
 * based on intel anv code:
 * Copyright © 2015 Intel Corporation
 */

#include "tu_wsi.h"

#include "drm-uapi/drm_fourcc.h"

#include "vk_util.h"
#include "wsi_common_drm.h"

#include "tu_device.h"

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
tu_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(tu_physical_device, pdevice, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(&pdevice->instance->vk, pName);
}

static bool
tu_wsi_can_present_on_device(VkPhysicalDevice physicalDevice, int fd)
{
#ifdef HAVE_LIBDRM
   VK_FROM_HANDLE(tu_physical_device, pdevice, physicalDevice);
   return wsi_common_drm_devices_equal(fd, pdevice->local_fd);
#else
   return true;
#endif
}

VkResult
tu_wsi_init(struct tu_physical_device *physical_device)
{
   VkResult result;

   /* The vDRM Windows guest has no D3D12 queue exposed to Mesa.  Use the
    * Win32 DIB path so surface/swapchain/present can operate without DXGI
    * composition; GPU rendering remains exercised by the Vulkan device. */
   const struct wsi_device_options options = {
#ifdef VK_USE_PLATFORM_WIN32_KHR
      .sw_device = true,
#else
      .sw_device = false,
#endif
   };
   result = wsi_device_init(&physical_device->wsi_device,
                            tu_physical_device_to_handle(physical_device),
                            tu_wsi_proc_addr,
                            &physical_device->instance->vk.alloc,
                            physical_device->master_fd,
                            &physical_device->instance->drirc.options,
                            &options);
   if (result != VK_SUCCESS)
      return result;

#ifdef VK_USE_PLATFORM_WIN32_KHR
   /* The GDI present path needs the pixels in a host-visible linear
    * allocation. By default the common WSI code picks
    * WSI_SWAPCHAIN_BUFFER_BLIT: the app renders a tiled+UBWC swapchain image
    * and the pre-present submit copies it into a linear buffer with one
    * 2D-engine pass (vkCmdCopyImageToBuffer). The GDI present thread waits
    * for that submit instead of the render thread (see wsi_common_win32.cpp),
    * so the extra guest->host->GPU->fence round trip no longer sits on the
    * app's critical path.
    *
    * TU_WSI_LINEAR=1 renders a VK_IMAGE_TILING_LINEAR swapchain image directly
    * and skips the blit submit. That path hangs on some GPUs of the same
    * generation (root cause not yet identified), so it stays opt-in for A/B
    * runs only. */
   {
      const char *env = getenv("TU_WSI_LINEAR");
      if (env && *env && *env != '0')
         physical_device->wsi_device.wants_linear = true;
   }
#endif

   physical_device->wsi_device.supports_modifiers = true;
   physical_device->wsi_device.can_present_on_device =
      tu_wsi_can_present_on_device;

   physical_device->vk.wsi_device = &physical_device->wsi_device;

   return VK_SUCCESS;
}

void
tu_wsi_finish(struct tu_physical_device *physical_device)
{
   physical_device->vk.wsi_device = NULL;
   wsi_device_finish(&physical_device->wsi_device,
                     &physical_device->instance->vk.alloc);
}
