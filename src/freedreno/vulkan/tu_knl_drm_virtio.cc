/*
 * Copyright © 2018 Google, Inc.
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Kernel interface layer for turnip running on virtio_gpu (aka virtgpu)
 */

#include "drm-uapi/msm_drm.h"
#include "drm-uapi/virtgpu_drm.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#if !defined(_WIN32)
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <xf86drm.h>
#else
#include <windows.h>
/* sys/mman.h 在 mingw 不存在；vdrm 后端的失败约定用 (void *)-1 保持一致，
 * 由 vdrm_wddm.c（块 2+）实现真正的 CPU map 语义。 */
#define MAP_FAILED ((void *)-1)
#endif

#include "util/hash_table.h"
#if !defined(_WIN32)
#include "util/libsync.h"
#endif
#include "util/u_debug.h"
#include "util/u_process.h"
#include "vk_drm_syncobj.h"
#include "vk_util.h"

#include "tu_cmd_buffer.h"
#include "tu_cs.h"
#include "tu_device.h"
#include "tu_dynamic_rendering.h"
#include "tu_knl.h"
#include "tu_knl_drm.h"
#include "tu_queue.h"

/* NOLINTBEGIN */
/* clang-format off */
#include "vdrm.h"
#include "msm_proto.h"
/* clang-format on */
/* NOLINTEND */

/* Windows guest BOs use MmCached mappings and KGSL IOCOHERENT imports.
 * Publish/observe coherent fence memory without entering cache_ops_null.
 * Linux guest-pool backing still requires its explicit cache maintenance.
 */
static void
virtio_sync_guest_bo_cache(struct tu_device *dev, struct tu_bo *bo,
                          VkDeviceSize offset, VkDeviceSize size,
                          enum tu_mem_sync_op op)
{
#ifdef _WIN32
   __atomic_thread_fence(__ATOMIC_SEQ_CST);
#else
   tu_bo_sync_cache(dev, bo, offset, size, op);
#endif
}

#if defined(_WIN32)
/* vk_drm_syncobj.c 只在 libdrm 可用时编入；Windows 上没有任何真实的 drm
 * syncobj，vulkan runtime 侧也走不到。这里给一个占位实现，仅让
 * vk_sync_as_drm_syncobj() 的内联 finish 指针比较可链接。 */
extern "C" void
vk_drm_syncobj_finish(struct vk_device *device, struct vk_sync *sync)
{
   (void) device;
   (void) sync;
}
#endif

#if defined(_WIN32)
/*
 * Linux <asm-generic/ioctl.h> 的 _IOC 编码。msm_proto 的 IOCTL_SIMPLE ccmd
 * 里携带的是内核 DRM_IOCTL_* 编号（宿主 drm2kgsl 按 Linux 规则解码），mingw
 * 没有这套宏，这里按 asm-generic（与 Linux arm64 相同、与架构无关的编码）
 * 原样补一份，保证 DRM_IOCTL_MSM_* 常量值两边一致。
 */
#ifndef _IOC_NRBITS
#define _IOC_NRBITS 8
#define _IOC_TYPEBITS 8
#define _IOC_SIZEBITS 14
#define _IOC_DIRBITS 2

#define _IOC_NRMASK ((1 << _IOC_NRBITS) - 1)
#define _IOC_TYPEMASK ((1 << _IOC_TYPEBITS) - 1)
#define _IOC_SIZEMASK ((1 << _IOC_SIZEBITS) - 1)
#define _IOC_DIRMASK ((1 << _IOC_DIRBITS) - 1)

#define _IOC_NRSHIFT 0
#define _IOC_TYPESHIFT (_IOC_NRSHIFT + _IOC_NRBITS)
#define _IOC_SIZESHIFT (_IOC_TYPESHIFT + _IOC_TYPEBITS)
#define _IOC_DIRSHIFT (_IOC_SIZESHIFT + _IOC_SIZEBITS)

#define _IOC_NONE 0U
#define _IOC_WRITE 1U
#define _IOC_READ 2U

#define _IOC(dir, type, nr, size) \
   (((dir) << _IOC_DIRSHIFT) | ((type) << _IOC_TYPESHIFT) | \
    ((nr) << _IOC_NRSHIFT) | ((size) << _IOC_SIZESHIFT))

#define _IO(type, nr) _IOC(_IOC_NONE, (type), (nr), 0)
#define _IOR(type, nr, size) _IOC(_IOC_READ, (type), (nr), sizeof(size))
#define _IOW(type, nr, size) _IOC(_IOC_WRITE, (type), (nr), sizeof(size))
#define _IOWR(type, nr, size) \
   _IOC(_IOC_READ | _IOC_WRITE, (type), (nr), sizeof(size))

#define _IOC_DIR(nr) (((nr) >> _IOC_DIRSHIFT) & _IOC_DIRMASK)
#define _IOC_TYPE(nr) (((nr) >> _IOC_TYPESHIFT) & _IOC_TYPEMASK)
#define _IOC_NR(nr) (((nr) >> _IOC_NRSHIFT) & _IOC_NRMASK)
#define _IOC_SIZE(nr) (((nr) >> _IOC_SIZESHIFT) & _IOC_SIZEMASK)

#define IOC_IN (_IOC_WRITE << _IOC_DIRSHIFT)
#define IOC_OUT (_IOC_READ << _IOC_DIRSHIFT)
#endif
#endif

struct tu_userspace_fence_cmd {
   uint32_t pkt[4];    /* first 4 dwords of packet */
   uint32_t fence;     /* fifth dword is fence value which is plugged in at runtime */
   uint32_t _pad[11];
};

struct tu_userspace_fence_cmds {
   struct tu_userspace_fence_cmd cmds[64];
};

struct tu_virtio_device {
   struct vdrm_device *vdrm;
   struct msm_shmem *shmem;
   uint32_t next_blob_id;

   struct tu_userspace_fence_cmds *fence_cmds;
   struct tu_bo *fence_cmds_mem;

   /* Holds the fence of the most recent real submit (appended as an extra
    * out-syncobj on every EXECBUFFER).  Empty submits (e.g. the WSI present
    * submit: no commands, semaphore-only) are then resolved guest-locally by
    * SYNCOBJ_TRANSFER from this instead of a host round-trip.  0 if
    * unavailable (vtest, create failure) which disables the fast path. */
   uint32_t last_submit_syncobj;

   /**
    * Processing zombie VMAs is a two step process, first we clear the iova
    * and then we close the handles.  But to minimize waste of virtqueue
    * space (and associated stalling and ping-ponging between guest and host)
    * we want to batch up all the GEM_SET_IOVA ccmds before we flush them to
    * the host and start closing handles.
    *
    * This gives us a place to stash the VMAs between the two steps.
    */
   struct u_vector zombie_vmas_stage_2;
};

static int tu_drm_get_param(struct vdrm_device *vdrm, uint32_t param, uint64_t *value);

/**
 * Helper for simple pass-thru ioctls
 */
static int
virtio_simple_ioctl(struct vdrm_device *vdrm, unsigned cmd, void *_req)
{
   MESA_TRACE_FUNC();
   unsigned req_len = sizeof(struct msm_ccmd_ioctl_simple_req);
   unsigned rsp_len = sizeof(struct msm_ccmd_ioctl_simple_rsp);

   req_len += _IOC_SIZE(cmd);
   if (cmd & IOC_OUT)
      rsp_len += _IOC_SIZE(cmd);

   uint8_t buf[req_len];
   struct msm_ccmd_ioctl_simple_req *req = (struct msm_ccmd_ioctl_simple_req *)buf;
   struct msm_ccmd_ioctl_simple_rsp *rsp;

   req->hdr = MSM_CCMD(IOCTL_SIMPLE, req_len);
   req->cmd = cmd;
   memcpy(req->payload, _req, _IOC_SIZE(cmd));

   rsp = (struct msm_ccmd_ioctl_simple_rsp *)
         vdrm_alloc_rsp(vdrm, &req->hdr, rsp_len);

   int ret = vdrm_send_req(vdrm, &req->hdr, true);

   if (cmd & IOC_OUT)
      memcpy(_req, rsp->payload, _IOC_SIZE(cmd));

   ret = rsp->ret;

   return ret;
}

static int
set_iova(struct tu_device *device, uint32_t res_id, uint64_t iova)
{
   struct msm_ccmd_gem_set_iova_req req = {
         .hdr = MSM_CCMD(GEM_SET_IOVA, sizeof(req)),
         .iova = iova,
         .res_id = res_id,
   };

   return vdrm_send_req(device->vdev->vdrm, &req.hdr, false);
}

static int
query_faults(struct tu_device *dev, uint64_t *value)
{
   struct tu_virtio_device *vdev = dev->vdev;
   uint32_t async_error = 0;
   uint64_t global_faults;

   if (vdrm_shmem_has_field(vdev->shmem, async_error))
      async_error = vdev->shmem->async_error;

   if (vdrm_shmem_has_field(vdev->shmem, global_faults)) {
      global_faults = vdev->shmem->global_faults;
   } else {
      int ret = tu_drm_get_param(vdev->vdrm, MSM_PARAM_FAULTS, &global_faults);
      if (ret)
         return ret;
   }

   *value = global_faults + async_error;

   return 0;
}

static void
set_debuginfo(struct tu_device *dev)
{
   const char *comm = util_get_process_name();
   static char cmdline[0x1000];

   if (!comm || !util_get_command_line(cmdline, sizeof(cmdline)))
      return;

   unsigned comm_len = strlen(comm) + 1;
   unsigned cmdline_len = strlen(cmdline) + 1;

   struct msm_ccmd_set_debuginfo_req *req;

   unsigned req_len = align(sizeof(*req) + comm_len + cmdline_len, 4);

   req = (struct msm_ccmd_set_debuginfo_req *)malloc(req_len);

   req->hdr         = MSM_CCMD(SET_DEBUGINFO, req_len);
   req->comm_len    = comm_len;
   req->cmdline_len = cmdline_len;

   memcpy(&req->payload[0], comm, comm_len);
   memcpy(&req->payload[comm_len], cmdline, cmdline_len);

   vdrm_send_req(dev->vdev->vdrm, &req->hdr, false);

   free(req);
}

static VkResult
virtio_device_init(struct tu_device *dev)
{
   struct tu_instance *instance = dev->physical_device->instance;
   int fd;

#if defined(_WIN32)
   /* Windows guest：没有 /dev/dri 节点；vdrm_wddm 后端直接从 D3DKMT 枚举
    * 适配器（块 2），fd 概念不存在，恒 -1。 */
   fd = -1;
#else
   if (strlen(dev->physical_device->fd_path) == 0) {
      fd = -1;
   } else {
      fd = open(dev->physical_device->fd_path, O_RDWR | O_CLOEXEC);
      if (fd < 0) {
         return vk_startup_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                                 "failed to open device %s", dev->physical_device->fd_path);
      }
   }
#endif

   struct tu_virtio_device *vdev = (struct tu_virtio_device *)
            vk_zalloc(&instance->vk.alloc, sizeof(*vdev), 8,
                      VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!vdev) {
      close(fd);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   };

   u_vector_init(&vdev->zombie_vmas_stage_2, 64, sizeof(struct tu_zombie_vma));

   dev->vdev = vdev;
   dev->fd = fd;

   vdev->vdrm = vdrm_device_connect(fd, VIRTGPU_DRM_CONTEXT_MSM);
   if (!vdev->vdrm) {
      u_vector_finish(&vdev->zombie_vmas_stage_2);
      vk_free(&instance->vk.alloc, vdev);
      dev->vdev = NULL;
      close(fd);
      return vk_startup_errorf(instance, VK_ERROR_INITIALIZATION_FAILED,
                               "failed to connect virtio DRM context");
   }

#ifdef _WIN32
   /* Shared D3D11 resources must be allocated by the UMD's pfnAllocateCb, not
    * by the transport's own device -- only the runtime mints the kernel resource
    * handle an application can share.  Absent when the ICD runs without a D3D
    * UMD, which just means no shared resource support. */
   if (instance->runtime_alloc_fn && instance->runtime_free_fn) {
      vdrm_wddm_set_runtime_allocator(vdev->vdrm, instance->runtime_alloc_ctx,
                                      instance->runtime_alloc_fn,
                                      instance->runtime_free_fn);
   }
#endif

   /* The physical-device probe uses a short-lived vdrm context.  Query the
    * slice again on the real VkDevice context, otherwise every device keeps
    * allocating from the probe context's slice and defeats host isolation. */
   uint64_t va_start = 0, va_size = 0;
   if (!tu_drm_get_param(vdev->vdrm, MSM_PARAM_VA_START, &va_start) &&
       va_start &&
       !tu_drm_get_param(vdev->vdrm, MSM_PARAM_VA_SIZE, &va_size) && va_size) {
      dev->va_start = va_start;
      dev->va_size = va_size;
   }

#if !defined(_WIN32)
   if (fd >= 0 && drmSyncobjCreate(fd, 0, &vdev->last_submit_syncobj))
      vdev->last_submit_syncobj = 0;
#endif

   p_atomic_set(&vdev->next_blob_id, (uint32_t)(getpid() & 0x7fff) << 16);
   vdev->shmem = to_msm_shmem(vdev->vdrm->shmem);

   query_faults(dev, &dev->fault_count);

   set_debuginfo(dev);

#if defined(_WIN32)
   /* TODO(块 3/5): vdrm_wddm 的 fence/sync 模型（ResourceBusy event / fence
    * id），现在连 connect 都失败，运行期到不了这里。 */
   (void) vdev;
#else
   if (fd < 0)
      dev->vk.sync = vdrm_vpipe_get_sync(vdev->vdrm);

   if (fd >= 0)
      dev->vk.copy_sync_payloads = vk_drm_syncobj_copy_payloads;
#endif

   return VK_SUCCESS;
}

static void
virtio_device_finish(struct tu_device *dev)
{
   struct tu_instance *instance = dev->physical_device->instance;
   struct tu_virtio_device *vdev = dev->vdev;

   u_vector_finish(&vdev->zombie_vmas_stage_2);

#if !defined(_WIN32)
   if (vdev->last_submit_syncobj)
      drmSyncobjDestroy(dev->fd, vdev->last_submit_syncobj);
#endif

   vdrm_device_close(vdev->vdrm);

   vk_free(&instance->vk.alloc, vdev);
   dev->vdev = NULL;

#if !defined(_WIN32)
   close(dev->fd);
#else
   dev->fd = -1;
#endif
}

static int
tu_drm_get_param(struct vdrm_device *vdrm, uint32_t param, uint64_t *value)
{
   MESA_TRACE_FUNC();

   /* Technically this requires a pipe, but the kernel only supports one pipe
    * anyway at the time of writing and most of these are clearly pipe
    * independent. */
   struct drm_msm_param req = {
      .pipe = MSM_PIPE_3D0,
      .param = param,
   };

   int ret = virtio_simple_ioctl(vdrm, DRM_IOCTL_MSM_GET_PARAM, &req);
   if (ret)
      return ret;

   *value = req.value;

   return 0;
}

static uint32_t
tu_drm_get_highest_bank_bit(struct vdrm_device *vdrm)
{
   uint64_t value;
   int ret = tu_drm_get_param(vdrm, MSM_PARAM_HIGHEST_BANK_BIT, &value);
   if (ret)
      return 0;

   return value;
}

static enum fdl_macrotile_mode
tu_drm_get_macrotile_mode(struct vdrm_device *vdrm)
{
   if (vdrm->caps.u.msm.macrotile_mode)
      return (enum fdl_macrotile_mode)vdrm->caps.u.msm.macrotile_mode;
   uint64_t value;
   int ret = tu_drm_get_param(vdrm, MSM_PARAM_MACROTILE_MODE, &value);
   if (ret)
      return FDL_MACROTILE_INVALID;

   return (enum fdl_macrotile_mode) value;
}

static uint32_t
tu_drm_get_ubwc_swizzle(struct vdrm_device *vdrm)
{
   if (vdrm->caps.u.msm.ubwc_swizzle)
      return (uint32_t)vdrm->caps.u.msm.ubwc_swizzle;
   uint64_t value;
   int ret = tu_drm_get_param(vdrm, MSM_PARAM_UBWC_SWIZZLE, &value);
   if (ret)
      return ~0;

   return value;
}

static uint64_t
tu_drm_get_uche_trap_base(struct vdrm_device *vdrm)
{
   uint64_t value;
   int ret = tu_drm_get_param(vdrm, MSM_PARAM_UCHE_TRAP_BASE, &value);
   if (ret)
      return 0x1fffffffff000ull;

   return value;
}

static int
virtio_device_get_gpu_timestamp(struct tu_device *dev, uint64_t *ts)
{
   MESA_TRACE_FUNC();
   return tu_drm_get_param(dev->vdev->vdrm, MSM_PARAM_TIMESTAMP, ts);
}

static int
virtio_device_get_suspend_count(struct tu_device *dev, uint64_t *suspend_count)
{
   int ret = tu_drm_get_param(dev->vdev->vdrm, MSM_PARAM_SUSPENDS, suspend_count);
   return ret;
}

static bool
opt_cap_bool(uint32_t val)
{
   if (val == VIRTGPU_CAP_BOOL_TRUE)  return true;
   if (val == VIRTGPU_CAP_BOOL_FALSE) return false;
   UNREACHABLE("invalid val");
}

static bool
tu_drm_get_raytracing(struct vdrm_device *vdrm)
{
   if (vdrm->caps.u.msm.has_raytracing)
      return opt_cap_bool(vdrm->caps.u.msm.has_raytracing);
   uint64_t value;
   int ret = tu_drm_get_param(vdrm, MSM_PARAM_RAYTRACING, &value);
   if (ret)
      return false;

   return value;
}


static VkResult
virtio_device_check_status(struct tu_device *device)
{
   uint64_t last_fault_count = device->fault_count;

   query_faults(device, &device->fault_count);

   if (last_fault_count != device->fault_count)
      return vk_device_set_lost(&device->vk, "GPU faulted or hung");

   return VK_SUCCESS;
}

static int
virtio_submitqueue_new(struct tu_device *dev, struct tu_queue *queue)
{
   MESA_TRACE_FUNC();

   assert(queue->priority >= 0 &&
          queue->priority < dev->physical_device->submitqueue_priority_count);

   struct drm_msm_submitqueue req = {
      .flags = queue->type == TU_QUEUE_SPARSE ? MSM_SUBMITQUEUE_VM_BIND :
         (dev->physical_device->info->chip >= 7 &&
          dev->physical_device->has_preemption ?
          MSM_SUBMITQUEUE_ALLOW_PREEMPT : 0),
      .prio = queue->priority,
   };

   int ret = virtio_simple_ioctl(dev->vdev->vdrm, DRM_IOCTL_MSM_SUBMITQUEUE_NEW, &req);
   if (ret)
      return ret;

   queue->msm_queue_id = req.id;
   return 0;
}

static void
virtio_submitqueue_close(struct tu_device *dev, struct tu_queue *queue)
{
   MESA_TRACE_FUNC();
   virtio_simple_ioctl(dev->vdev->vdrm, DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE, &queue->msm_queue_id);
}

static bool
virtio_has_preemption(struct vdrm_device *vdrm)
{
   if (vdrm->caps.u.msm.has_preemption)
      return opt_cap_bool(vdrm->caps.u.msm.has_preemption);
   struct drm_msm_submitqueue req = {
      .flags = MSM_SUBMITQUEUE_ALLOW_PREEMPT,
      .prio = vdrm->caps.u.msm.priorities / 2,
   };

   int ret = virtio_simple_ioctl(vdrm, DRM_IOCTL_MSM_SUBMITQUEUE_NEW, &req);
   if (ret)
      return false;

   virtio_simple_ioctl(vdrm, DRM_IOCTL_MSM_SUBMITQUEUE_CLOSE, &req.id);
   return true;
}

static VkResult
tu_wait_fence(struct tu_device *dev,
              uint32_t queue_id,
              int fence,
              uint64_t timeout_ns)
{
   struct vdrm_device *vdrm = dev->vdev->vdrm;
   MESA_TRACE_FUNC();

   if (!fence_before(dev->global_bo_map->userspace_fence, fence))
      return VK_SUCCESS;

   if (!timeout_ns)
      return VK_TIMEOUT;

   MESA_TRACE_FUNC();

   struct msm_ccmd_wait_fence_req req = {
         .hdr = MSM_CCMD(WAIT_FENCE, sizeof(req)),
         .queue_id = queue_id,
         .fence = fence,
   };
   struct msm_ccmd_submitqueue_query_rsp *rsp;
   int64_t end_time = os_time_get_nano() + timeout_ns;
   int ret;

   do {
      rsp = (struct msm_ccmd_submitqueue_query_rsp *)
            vdrm_alloc_rsp(vdrm, &req.hdr, sizeof(*rsp));

      ret = vdrm_send_req(vdrm, &req.hdr, true);
      if (ret)
         goto out;

      ret = rsp->ret;

      if (timeout_ns != OS_TIMEOUT_INFINITE &&
          os_time_get_nano() >= end_time)
         break;
   } while (ret == -ETIMEDOUT);

out:
   if (!ret) return VK_SUCCESS;
   if (ret == -ETIMEDOUT) return VK_TIMEOUT;
   return VK_ERROR_UNKNOWN;
}

static VkResult
virtio_queue_wait_fence(struct tu_queue *queue, uint32_t fence,
                        uint64_t timeout_ns)
{
   MESA_TRACE_FUNC();
   return tu_wait_fence(queue->device, queue->msm_queue_id, fence,
                        timeout_ns);
}

/* Poll-first client sync waits
 *
 * Every submit already makes the GPU write its seqno to
 * global_bo->userspace_fence (see setup_fence_cmds), and tu-internal waits
 * use that via tu_wait_fence().  Client fence/semaphore waits however go
 * through vk_drm_syncobj -> DRM_IOCTL_SYNCOBJ_WAIT -> a real sleep that is
 * only woken once the host completion (fence poll thread -> add_used ->
 * completion vIRQ -> guest dma_fence signal) has run its course, which on
 * the virtio/KGSL native context costs a wakeup chain per frame.
 *
 * tu_virtio_sync wraps vk_drm_syncobj: the submit path records which queue
 * seqno will signal the sync, and the wait entrypoint first polls the
 * userspace fence (plus a short bounded spin) before falling back to the
 * regular syncobj wait.  The GPU write happens at the end of the same
 * cmdstream that the syncobj's dma_fence completes with, so "userspace fence
 * reached the seqno" implies the submit's GPU work is done - the only thing
 * skipped is the host->guest signal plumbing, which no longer needs to be
 * waited for.
 *
 * Every payload-changing entrypoint (reset/move/import/CPU signal)
 * invalidates the record, so dma-buf-imported payloads (WSI acquire fences)
 * and anything else not signaled by our own submits always take the
 * fallback.  The record is also only written while the device has a single
 * queue: userspace_fence is one device-global slot, so with multiple queues
 * writing it the compare would be meaningless.
 *
 * .finish is deliberately kept as the base vk_drm_syncobj_finish so that
 * vk_sync_type_is_drm_syncobj() (and thus vk_sync_as_drm_syncobj()) keeps
 * treating these syncs as plain drm syncobjs, e.g. on the submit path.
 */

struct tu_virtio_sync {
   struct vk_drm_syncobj drm; /* must be first */
   /* seqno of the queue submit that will signal this sync; valid only while
    * owner is non-NULL.  Written on the submit path, read by waiters; the
    * client's external-sync rules order those accesses. */
   struct tu_queue *owner;
   uint32_t submit_seqno;
#ifdef _WIN32
   /* Windows 没有 drm syncobj（drm.syncobj 恒 0），二元状态存这里：
    * true = 已确认完成（CPU signal 过，或等到过 userspace_fence）。
    * 见 tu_win_sync_type。 */
   bool signaled;
#endif
};

static inline struct tu_virtio_sync *
to_tu_virtio_sync(struct vk_sync *sync)
{
   return (struct tu_virtio_sync *)sync;
}

static inline void
tu_virtio_sync_invalidate(struct vk_sync *sync)
{
   p_atomic_set(&to_tu_virtio_sync(sync)->owner, (struct tu_queue *)NULL);
}

static const struct vk_sync_type *
tu_virtio_sync_base_type(struct vk_device *device)
{
   struct tu_device *dev = container_of(device, struct tu_device, vk);
   return &dev->physical_device->syncobj_type;
}

static VkResult
tu_virtio_sync_signal(struct vk_device *device, struct vk_sync *sync,
                      uint64_t value)
{
   tu_virtio_sync_invalidate(sync);
   return tu_virtio_sync_base_type(device)->signal(device, sync, value);
}

static VkResult
tu_virtio_sync_signal_many(struct vk_device *device, uint32_t signal_count,
                           const struct vk_sync_signal *signals)
{
   for (uint32_t i = 0; i < signal_count; i++)
      tu_virtio_sync_invalidate(signals[i].sync);
   return tu_virtio_sync_base_type(device)->signal_many(device, signal_count,
                                                        signals);
}

static VkResult
tu_virtio_sync_reset(struct vk_device *device, struct vk_sync *sync)
{
   tu_virtio_sync_invalidate(sync);
   return tu_virtio_sync_base_type(device)->reset(device, sync);
}

static VkResult
tu_virtio_sync_reset_many(struct vk_device *device, uint32_t sync_count,
                          struct vk_sync *const *syncs)
{
   for (uint32_t i = 0; i < sync_count; i++)
      tu_virtio_sync_invalidate(syncs[i]);
   return tu_virtio_sync_base_type(device)->reset_many(device, sync_count,
                                                       syncs);
}

static VkResult
tu_virtio_sync_move(struct vk_device *device, struct vk_sync *dst,
                    struct vk_sync *src)
{
   /* dst takes src's payload (which our record doesn't describe) and src is
    * reset; neither record survives. */
   tu_virtio_sync_invalidate(dst);
   if (src->type == dst->type)
      tu_virtio_sync_invalidate(src);
   return tu_virtio_sync_base_type(device)->move(device, dst, src);
}

static VkResult
tu_virtio_sync_import_opaque_fd(struct vk_device *device,
                                struct vk_sync *sync, int fd)
{
   tu_virtio_sync_invalidate(sync);
   return tu_virtio_sync_base_type(device)->import_opaque_fd(device, sync, fd);
}

static VkResult
tu_virtio_sync_import_sync_file(struct vk_device *device,
                                struct vk_sync *sync, int sync_file)
{
   tu_virtio_sync_invalidate(sync);
   return tu_virtio_sync_base_type(device)->import_sync_file(device, sync,
                                                             sync_file);
}

static int64_t
tu_poll_spin_ns(void)
{
   /* The spin window needs to cover the time from wait-start to the GPU's
    * userspace-fence write, or the spin is wasted AND the full syncobj-sleep
    * wakeup chain is paid on top - the behavior is all-or-nothing per scene.
    * Sweep on vkmark (FPS at 75/150/200/300/500 us):
    *   clear  2934 / 4108 / 8848 / 9142 / 8998
    *   vertex 5441 / 5969 / 9301 / 10434 / 10495
    * and 300us also covers the ~200us frames of effect2d (5680 -> 9747) that
    * 200us misses; beyond 300us it is flat.  Full-suite score 7783 (200us)
    * vs 9462 (300us).  Worst case a missed wait burns the whole window
    * before sleeping, which at GPU-bound frame times is noise. */
   static int64_t spin_ns = -1;
   if (spin_ns < 0)
      spin_ns = (int64_t)debug_get_num_option("TU_POLL_SPIN_US", 300) * 1000;
   return spin_ns;
}

static VkResult
tu_virtio_sync_wait_many(struct vk_device *device, uint32_t wait_count,
                         const struct vk_sync_wait *waits,
                         enum vk_sync_wait_flags wait_flags,
                         uint64_t abs_timeout_ns)
{
   struct tu_device *dev = container_of(device, struct tu_device, vk);
   const struct vk_sync_type *base = &dev->physical_device->syncobj_type;
   const struct vk_sync_type *poll_type = &dev->physical_device->poll_sync_type;

   if ((wait_flags & VK_SYNC_WAIT_PENDING) || wait_count == 0 ||
       !dev->global_bo_map)
      return base->wait_many(device, wait_count, waits, wait_flags,
                             abs_timeout_ns);

   /* Only take the fast path when every wait maps to a recorded seqno; a
    * single unpollable entry sends the whole wait down the fallback (for
    * WAIT_ANY a pollable subset could legally satisfy the wait early, but
    * mixed waits are not a per-frame pattern worth the complexity). */
   for (uint32_t i = 0; i < wait_count; i++) {
      if (waits[i].sync->type != poll_type ||
          !p_atomic_read(&to_tu_virtio_sync(waits[i].sync)->owner))
         return base->wait_many(device, wait_count, waits, wait_flags,
                                abs_timeout_ns);
   }

   int64_t spin_end = os_time_get_nano() + tu_poll_spin_ns();
   if (abs_timeout_ns < INT64_MAX && (int64_t)abs_timeout_ns < spin_end)
      spin_end = (int64_t)abs_timeout_ns;

   for (;;) {
      if (dev->vdev->vdrm->supports_guest_alloc) {
         virtio_sync_guest_bo_cache(dev, dev->global_bo, gb_offset(userspace_fence),
                          sizeof(dev->global_bo_map->userspace_fence),
                          TU_MEM_SYNC_CACHE_FROM_GPU);
      }
      uint32_t cur = dev->global_bo_map->userspace_fence;
      bool all = true, any = false;

      for (uint32_t i = 0; i < wait_count; i++) {
         if (!fence_before(cur, to_tu_virtio_sync(waits[i].sync)->submit_seqno))
            any = true;
         else
            all = false;
      }

      if ((wait_flags & VK_SYNC_WAIT_ANY) ? any : all)
         return VK_SUCCESS;

      if (os_time_get_nano() >= spin_end)
         break;

#ifdef __aarch64__
      __asm__ volatile("yield");
#endif
   }

   return base->wait_many(device, wait_count, waits, wait_flags,
                          abs_timeout_ns);
}

static struct vk_sync_type
tu_virtio_get_poll_sync_type(const struct vk_sync_type *base)
{
   struct vk_sync_type type = *base;

   type.size = sizeof(struct tu_virtio_sync);
   /* Timeline points would need their own point->seqno mapping; timeline
    * semaphores keep using the base type instead. */
   type.features = (enum vk_sync_features)
      (type.features & ~VK_SYNC_FEATURE_TIMELINE);
   type.get_value = NULL;
#ifndef _WIN32
   /* On Linux, owner is only a cache of the kernel syncobj payload.  On
    * Windows it is part of the payload itself: invalidating src before
    * move would lose a pending signal permanently.  Keep the Windows base
    * operations, including its NULL signal_many/reset_many fallbacks.
    */
   type.signal = tu_virtio_sync_signal;
   type.signal_many = tu_virtio_sync_signal_many;
   type.reset = tu_virtio_sync_reset;
   type.reset_many = tu_virtio_sync_reset_many;
   type.move = tu_virtio_sync_move;
#endif
#ifdef _WIN32
   /* base（tu_win_sync_type）没有 fd 导入，这里也不能留非 NULL 的钩子：
    * vk_semaphore.c:47 / vk_fence.c:44 是"按 import_opaque_fd 是否为空"挑
    * sync 类型的，留着会挑中它然后在 base 的 NULL 指针上崩。 */
   type.import_opaque_fd = NULL;
   type.import_sync_file = NULL;
#else
   type.import_opaque_fd = tu_virtio_sync_import_opaque_fd;
   type.import_sync_file = tu_virtio_sync_import_sync_file;
#endif
   type.wait_many = tu_virtio_sync_wait_many;

   return type;
}

#ifdef _WIN32
DEBUG_GET_ONCE_BOOL_OPTION(tu_win_sync_debug, "TU_WIN_SYNC_DEBUG", false)

/* Windows 上没有 drm syncobj / sync_file，vk_drm_syncobj 那一套无从落地。这里给
 * VkFence 和二元 VkSemaphore 做一个自足的二元 sync：状态 = 「已确认完成」布尔 +
 * 「由哪次提交签发的 seqno」，等待就是拿 seqno 跟 turnip 自己的 userspace fence
 * （global_bo_map->userspace_fence，GPU 侧 CACHE_FLUSH_TS 写）比。
 *
 * 载体沿用 Linux 的 struct tu_virtio_sync（drm.syncobj 在 Windows 恒 0 不用），
 * 这样 tu_virtio_get_poll_sync_type() 那层包装在 Windows 上也能照常套上去：
 * size / owner / submit_seqno 全部对得上，它只覆盖若干 op 并回落到 base（本类型）。
 * timeline 语义不做（features 不含 TIMELINE），交给 vk_sync_timeline 模拟。 */

static VkResult
tu_win_sync_init(struct vk_device *device, struct vk_sync *base,
                 uint64_t initial_value)
{
   struct tu_virtio_sync *sync = to_tu_virtio_sync(base);

   (void)device;
   sync->drm.syncobj = 0;
   sync->submit_seqno = 0;
   sync->signaled = initial_value != 0;
   p_atomic_set(&sync->owner, (struct tu_queue *)NULL);

   return VK_SUCCESS;
}

static void
tu_win_sync_finish(struct vk_device *device, struct vk_sync *base)
{
   (void)device;
   (void)base;
}

static VkResult
tu_win_sync_signal(struct vk_device *device, struct vk_sync *base, uint64_t value)
{
   struct tu_virtio_sync *sync = to_tu_virtio_sync(base);

   (void)device;
   (void)value;
   sync->submit_seqno = 0;
   sync->signaled = true;
   p_atomic_set(&sync->owner, (struct tu_queue *)NULL);

   return VK_SUCCESS;
}

static VkResult
tu_win_sync_reset(struct vk_device *device, struct vk_sync *base)
{
   struct tu_virtio_sync *sync = to_tu_virtio_sync(base);

   (void)device;
   sync->submit_seqno = 0;
   sync->signaled = false;
   p_atomic_set(&sync->owner, (struct tu_queue *)NULL);

   return VK_SUCCESS;
}

static VkResult
tu_win_sync_move(struct vk_device *device, struct vk_sync *dst_base,
                 struct vk_sync *src_base)
{
   struct tu_virtio_sync *dst = to_tu_virtio_sync(dst_base);
   struct tu_virtio_sync *src = to_tu_virtio_sync(src_base);

   (void)device;
   dst->submit_seqno = src->submit_seqno;
   dst->signaled = src->signaled;
   p_atomic_set(&dst->owner, p_atomic_read(&src->owner));

   src->submit_seqno = 0;
   src->signaled = false;
   p_atomic_set(&src->owner, (struct tu_queue *)NULL);

   return VK_SUCCESS;
}

/* 已完成？signaled 是"确认过"的缓存，避免每次都读共享页。 */
static bool
tu_win_sync_poll(struct tu_device *dev, struct tu_virtio_sync *sync)
{
   if (sync->signaled)
      return true;
   if (!p_atomic_read(&sync->owner) || !dev->global_bo_map)
      return false;
   if (fence_before(dev->global_bo_map->userspace_fence, sync->submit_seqno))
      return false;

   sync->signaled = true;
   return true;
}

class tu_win_sync_timer {
public:
   ~tu_win_sync_timer()
   {
      if (timer)
         CloseHandle(timer);
   }

   void wait(uint64_t duration_ns)
   {
      if (!initialized) {
         timer = CreateWaitableTimerExW(NULL, NULL,
                                       CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                       TIMER_MODIFY_STATE | SYNCHRONIZE);
         initialized = true;
      }

      if (timer) {
         LARGE_INTEGER due;
         due.QuadPart = -(int64_t)DIV_ROUND_UP(duration_ns, 100);
         if (SetWaitableTimerEx(timer, &due, 0, NULL, NULL, NULL, 0) &&
             WaitForSingleObject(timer, INFINITE) == WAIT_OBJECT_0)
            return;

         CloseHandle(timer);
         timer = NULL;
      }

      /* Keep the original blocking fallback if high-resolution timers fail. */
      os_time_sleep(DIV_ROUND_UP(duration_ns, 1000));
   }

private:
   HANDLE timer = NULL;
   bool initialized = false;
};

static VkResult
tu_win_sync_wait_many(struct vk_device *_device, uint32_t wait_count,
                      const struct vk_sync_wait *waits,
                      enum vk_sync_wait_flags wait_flags,
                      uint64_t abs_timeout_ns)
{
   struct tu_device *dev = container_of(_device, struct tu_device, vk);

   if (wait_count == 0)
      return VK_SUCCESS;

   const bool debug = debug_get_option_tu_win_sync_debug();
   uint64_t next_log_ns = debug ? os_time_get_nano() + 1000000000ull : 0;
   tu_win_sync_timer timer;

   for (;;) {
      bool all = true, any = false;

      for (uint32_t i = 0; i < wait_count; i++) {
         struct tu_virtio_sync *sync = to_tu_virtio_sync(waits[i].sync);
         bool done;

         /* WAIT_PENDING 只问"有没有被提交出去"，不等真正完成。 */
         if (wait_flags & VK_SYNC_WAIT_PENDING)
            done = sync->signaled || p_atomic_read(&sync->owner) != NULL;
         else
            done = tu_win_sync_poll(dev, sync);

         if (done)
            any = true;
         else
            all = false;
      }

      if ((wait_flags & VK_SYNC_WAIT_ANY) ? any : all)
         return VK_SUCCESS;

      if (debug && (uint64_t)os_time_get_nano() >= next_log_ns) {
         struct tu_virtio_device *vdev = dev->vdev;
         fprintf(stderr, "tu-win32-sync: waiting flags=0x%x gpu=%u "
                 "host_seq=%u next_req=%u async_error=%u global_faults=%u\n",
                 (unsigned)wait_flags,
                 dev->global_bo_map ? dev->global_bo_map->userspace_fence : 0,
                 vdev->shmem->base.seqno, vdev->vdrm->next_seqno,
                 vdev->shmem->async_error, vdev->shmem->global_faults);
         for (uint32_t i = 0; i < wait_count; i++) {
            struct tu_virtio_sync *sync = to_tu_virtio_sync(waits[i].sync);
            fprintf(stderr, "tu-win32-sync: wait[%u] sync=%p owner=%p "
                    "fence=%u signaled=%u\n", i, (void *)sync,
                    (void *)p_atomic_read(&sync->owner), sync->submit_seqno,
                    (unsigned)sync->signaled);
         }
         next_log_ns = os_time_get_nano() + 1000000000ull;
      }

      const uint64_t now = os_time_get_nano();
      if (abs_timeout_ns == 0 ||
          (abs_timeout_ns != OS_TIMEOUT_INFINITE && now >= abs_timeout_ns))
         return VK_TIMEOUT;

      /* Sleep(1) can cost a whole scheduler tick on Windows. Reuse a timer
       * while polling, and never request a wait beyond the caller's deadline.
       * Timer expiry only prompts another fence check; it does not signal it.
       */
      const uint64_t remaining = abs_timeout_ns == OS_TIMEOUT_INFINITE ?
                                 UINT64_MAX : abs_timeout_ns - now;
      timer.wait(MIN2(remaining, 100000ull));
   }
}

static VkResult
tu_win_sync_wait(struct vk_device *device, struct vk_sync *base,
                 uint64_t wait_value, enum vk_sync_wait_flags wait_flags,
                 uint64_t abs_timeout_ns)
{
   struct vk_sync_wait wait = { .sync = base, .wait_value = wait_value };

   return tu_win_sync_wait_many(device, 1, &wait, wait_flags, abs_timeout_ns);
}

static const struct vk_sync_type tu_win_sync_type = {
   .size = sizeof(struct tu_virtio_sync),
   .features = (enum vk_sync_features)(VK_SYNC_FEATURE_BINARY |
                                       VK_SYNC_FEATURE_GPU_WAIT |
                                       VK_SYNC_FEATURE_CPU_WAIT |
                                       VK_SYNC_FEATURE_CPU_RESET |
                                       VK_SYNC_FEATURE_CPU_SIGNAL |
                                       VK_SYNC_FEATURE_WAIT_ANY |
                                       VK_SYNC_FEATURE_WAIT_PENDING),
   .init = tu_win_sync_init,
   .finish = tu_win_sync_finish,
   .signal = tu_win_sync_signal,
   .reset = tu_win_sync_reset,
   .move = tu_win_sync_move,
   .wait = tu_win_sync_wait,
   .wait_many = tu_win_sync_wait_many,
};
#endif /* _WIN32 */
static bool
tu_virtio_single_queue(struct tu_device *dev)
{
   unsigned queue_count = 0;
   for (unsigned i = 0; i < TU_MAX_QUEUE_FAMILIES; i++)
      queue_count += dev->queue_count[i];
   return queue_count == 1;
}

/* Empty-submit fast path
 *
 * The per-frame WSI present submit (wsi_common.c QueuePresent) carries no
 * command buffer: it waits the client's render-complete semaphores (same
 * queue, so already ordered) and signals the swapchain per-image fence plus
 * the dma_buf/present semaphores.  Pushing that through EXECBUFFER costs a
 * kick, a host GPU_COMMAND+TIMESTAMP_EVENT pair and a completion vIRQ per
 * frame - half of the entire per-frame host traffic.
 *
 * Since every real submit parks its fence in vdev->last_submit_syncobj, an
 * empty submit whose waits are all known-ordered behind this queue's already
 * submitted work can be resolved entirely guest-side: give each signal target
 * the previous submit's fence via SYNCOBJ_TRANSFER.  Signal-ordering
 * semantics are preserved (the transferred fence signals no earlier than any
 * same-queue wait could require; at worst it signals later than strictly
 * needed, which the poll-first fast path absorbs).
 *
 * That inheritance path needs every wait to be provably ordered behind this
 * queue's submitted work, which a WSI acquire fence (a dma-buf import) or a
 * cross-queue wait is not.  Such a submit is still command-less though, so
 * rather than fall all the way back to EXECBUFFER it takes a second tier:
 * tu_empty_submit_copy_payloads() hands each wait's own fence to the signal
 * targets with vk_drm_syncobj_copy_payloads(), which is a few guest-local
 * syncobj ioctls (TRANSFER, or export/SYNC_IOC_MERGE/import without timeline
 * syncobj support) and still no host round-trip.  It costs more ioctls than
 * the inheritance tier, hence the ordering, but far less than a kick plus a
 * host GPU_COMMAND+TIMESTAMP_EVENT pair plus a completion vIRQ.
 *
 * The tiers are therefore:
 *   1. inheritance from last_submit_syncobj  - one TRANSFER per signal
 *   2. copy_payloads from the waits          - a few guest-local ioctls
 *   3. real EXECBUFFER                       - commands/binds/u_trace only
 *
 * TU_NO_EMPTY_SUBMIT=1 disables all of this; TU_NO_EMPTY_SUBMIT_COPY=1
 * disables only tier 2 so the two can be measured apart.
 */

static bool
tu_empty_submit_disabled(void)
{
   static int no_skip = -1;
   if (no_skip < 0)
      no_skip = debug_get_bool_option("TU_NO_EMPTY_SUBMIT", false);
   return no_skip;
}

static bool
tu_empty_submit_copy_disabled(void)
{
   static int no_copy = -1;
   if (no_copy < 0)
      no_copy = debug_get_bool_option("TU_NO_EMPTY_SUBMIT_COPY", false);
   return no_copy;
}

static bool
tu_empty_submit_can_skip(struct tu_queue *queue,
                         struct tu_msm_queue_submit *submit,
                         struct vk_sync_wait *waits, uint32_t wait_count,
                         struct vk_sync_signal *signals, uint32_t signal_count,
                         struct tu_u_trace_submission_data *u_trace_data)
{
   if (tu_empty_submit_disabled())
      return false;

   struct tu_device *dev = queue->device;

   if (dev->fd < 0 || !dev->vdev->last_submit_syncobj)
      return false;

   /* last_submit_syncobj is device-global: with more than one queue it may
    * hold another queue's (unordered) fence, so only skip when this queue is
    * provably the only one. */
   if (!tu_virtio_single_queue(dev))
      return false;

   /* Nothing real submitted yet: no fence to inherit */
   if (queue->fence <= 0)
      return false;

   if (u_trace_data)
      return false;

   if (submit->commands.size || submit->binds.size)
      return false;

   /* All waits must be provably ordered behind already-submitted work of
    * this queue: recorded poll-type syncs signaled by this queue.  Anything
    * else (dma-buf imports, cross-queue, timeline waits) takes the normal
    * path. */
   const struct vk_sync_type *poll_type =
      &dev->physical_device->poll_sync_type;
   for (uint32_t i = 0; i < wait_count; i++) {
      if (waits[i].sync->type != poll_type)
         return false;
      if (p_atomic_read(&to_tu_virtio_sync(waits[i].sync)->owner) != queue)
         return false;
   }

   /* All signal targets must be drm syncobjs we can TRANSFER into */
   for (uint32_t i = 0; i < signal_count; i++) {
      if (!vk_sync_as_drm_syncobj(signals[i].sync))
         return false;
   }

   return true;
}

static VkResult
tu_empty_submit_inherit(struct tu_queue *queue,
                        struct vk_sync_signal *signals, uint32_t signal_count)
{
#ifdef _WIN32
   /* Windows 没有 drm syncobj TRANSFER；空提交快路径（块 5）要换成 wddm
    * fence 语义。现在编译期内不可达（fd<0 使 can_skip 恒 false）。 */
   (void) queue;
   (void) signals;
   (void) signal_count;
   return VK_ERROR_FEATURE_NOT_PRESENT;
#else
   struct tu_device *dev = queue->device;
   struct tu_virtio_device *vdev = dev->vdev;
   const struct vk_sync_type *poll_type =
      &dev->physical_device->poll_sync_type;
   bool single_queue = tu_virtio_single_queue(dev);

   for (uint32_t i = 0; i < signal_count; i++) {
      struct vk_drm_syncobj *dst = vk_sync_as_drm_syncobj(signals[i].sync);

      /* dst_point = signal_value handles both binary (0) and timeline
       * targets; binary->timeline transfer is supported by the kernel. */
      int ret = drmSyncobjTransfer(dev->fd, dst->syncobj,
                                   signals[i].signal_value,
                                   vdev->last_submit_syncobj, 0, 0);
      if (ret) {
         return vk_device_set_lost(&dev->vk,
                                   "empty-submit syncobj transfer failed: %m");
      }

      if (single_queue && signals[i].sync->type == poll_type) {
         struct tu_virtio_sync *s = to_tu_virtio_sync(signals[i].sync);
         s->submit_seqno = queue->fence;
         p_atomic_set(&s->owner, queue);
      }
   }

   return VK_SUCCESS;
#endif
}

static VkResult
tu_empty_submit_copy_payloads(struct tu_queue *queue,
                              struct vk_sync_wait *waits, uint32_t wait_count,
                              struct vk_sync_signal *signals, uint32_t signal_count)
{
   struct tu_device *dev = queue->device;
   const struct vk_sync_type *poll_type =
      &dev->physical_device->poll_sync_type;

   if (tu_empty_submit_disabled() || tu_empty_submit_copy_disabled())
      return VK_ERROR_FEATURE_NOT_PRESENT;

   if (dev->fd < 0 || !dev->vk.copy_sync_payloads)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* Tier 2: arbitrary waits, but still command-less. copy_payloads hands
    * each wait's own fence to the signal targets, all guest-local (a few
    * SYNCOBJ_TRANSFER ioctls, or export/SYNC_IOC_MERGE/import fallback).
    * Signal targets get fences from elsewhere rather than this queue's
    * last_submit_syncobj, so poll records must be invalidated. */
   VkResult result = dev->vk.copy_sync_payloads(&dev->vk, wait_count, waits,
                                                signal_count, signals);
   if (result != VK_SUCCESS)
      return result;

   /* Invalidate poll records: signals now hold fences unrelated to
    * queue->fence, so reading userspace_fence[s->submit_seqno] would be
    * wrong. Set owner to NULL so poll_wait falls back to syncobj query. */
   for (uint32_t i = 0; i < signal_count; i++) {
      if (signals[i].sync->type == poll_type) {
         struct tu_virtio_sync *s = to_tu_virtio_sync(signals[i].sync);
         p_atomic_set(&s->owner, NULL);
      }
   }

   return VK_SUCCESS;
}

static VkResult
tu_free_zombie_vma_locked(struct tu_device *dev, bool wait)
{
   struct tu_virtio_device *vdev = dev->vdev;
   MESA_TRACE_FUNC();

   if (!u_vector_length(&dev->zombie_vmas) &&
       !u_vector_length(&vdev->zombie_vmas_stage_2))
      return VK_SUCCESS;

   if (wait && u_vector_length(&dev->zombie_vmas)) {
      struct tu_zombie_vma *vma = (struct tu_zombie_vma *)
            u_vector_head(&dev->zombie_vmas);
      /* Wait for 3s (arbitrary timeout) */
      VkResult ret = tu_wait_fence(dev, dev->queues[0]->msm_queue_id,
                                   vma->fence, 3000000000);

      if (ret != VK_SUCCESS)
         return ret;
   }

   /* Clear the iova of all finished objects in first pass so the SET_IOVA
    * ccmd's can be buffered and sent together to the host.  *Then* delete
    * the handles.  This avoids filling up the virtqueue with tiny messages,
    * since each execbuf ends up needing to be page aligned.
    */
   int last_signaled_fence = -1;
   while (u_vector_length(&dev->zombie_vmas) > 0) {
      struct tu_zombie_vma *vma = (struct tu_zombie_vma *)
            u_vector_tail(&dev->zombie_vmas);
      if (vma->fence > last_signaled_fence) {
         VkResult ret =
            tu_wait_fence(dev, dev->queues[0]->msm_queue_id, vma->fence, 0);
         if (ret != VK_SUCCESS)
            break;

         last_signaled_fence = vma->fence;
      }

      u_vector_remove(&dev->zombie_vmas);

      if (vma->gem_handle) {
         set_iova(dev, vma->res_id, 0);

         struct tu_zombie_vma *vma2 =
            (struct tu_zombie_vma *) u_vector_add(&vdev->zombie_vmas_stage_2);

         *vma2 = *vma;
      }
   }

#if defined(_WIN32)
   /* Unlike virtgpu_bo_close(), the WDDM close path does not flush buffered
    * ccmds. Submit SET_IOVA before DestroyAllocation detaches its resource;
    * otherwise the host sees a cleanup command referencing an already closed
    * BO. On failure retain the handles instead of overtaking those commands.
    */
   if (u_vector_length(&vdev->zombie_vmas_stage_2) && vdrm_flush(vdev->vdrm))
      return VK_ERROR_UNKNOWN;
#endif

   /* And _then_ close the GEM handles: */
   while (u_vector_length(&vdev->zombie_vmas_stage_2) > 0) {
      struct tu_zombie_vma *vma = (struct tu_zombie_vma *)
            u_vector_remove(&vdev->zombie_vmas_stage_2);

      util_vma_heap_free(&dev->vma, vma->iova, vma->size);
      vdrm_bo_close(dev->vdev->vdrm, vma->gem_handle);
   }

   return VK_SUCCESS;
}

static bool
tu_restore_from_zombie_vma_locked(struct tu_device *dev,
                                  uint32_t gem_handle,
                                  uint64_t *iova)
{
   MESA_TRACE_FUNC();
   struct tu_zombie_vma *vma;
   u_vector_foreach (vma, &dev->zombie_vmas) {
      if (vma->gem_handle == gem_handle) {
         *iova = vma->iova;

         /* mark to skip later vdrm bo and iova cleanup */
         vma->gem_handle = 0;
         return true;
      }
   }

   return false;
}

static VkResult
virtio_allocate_userspace_iova_locked(struct tu_device *dev,
                                      uint32_t gem_handle,
                                      uint64_t size,
                                      uint64_t client_iova,
                                      enum tu_bo_alloc_flags flags,
                                      uint64_t *iova)
{
   VkResult result;
   MESA_TRACE_FUNC();

   *iova = 0;

   if (flags & TU_BO_ALLOC_DMABUF) {
      assert(gem_handle);

      if (tu_restore_from_zombie_vma_locked(dev, gem_handle, iova))
         return VK_SUCCESS;
   }

   tu_free_zombie_vma_locked(dev, false);

   result = tu_allocate_userspace_iova(dev, size, client_iova, flags, iova);
   if (result == VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS) {
      /* Address may be already freed by us, but not considered as
       * freed by the kernel. We have to wait until all work that
       * may hold the address is done. Since addresses are meant to
       * be replayed only by debug tooling, it should be ok to wait.
       */
      tu_free_zombie_vma_locked(dev, true);
      result = tu_allocate_userspace_iova(dev, size, client_iova, flags, iova);
   }

   return result;
}

static VkResult
tu_bo_init(struct tu_device *dev,
           struct vk_object_base *base,
           struct tu_bo *bo,
           uint32_t gem_handle,
           uint64_t size,
           uint64_t iova,
           enum tu_bo_alloc_flags flags,
           const char *name)
{
   assert(dev->physical_device->has_set_iova);
   MESA_TRACE_FUNC();

   name = tu_debug_bos_add(dev, size, name);

   mtx_lock(&dev->bo_mutex);
   uint32_t idx = dev->submit_bo_count++;

   /* grow the bo list if needed */
   if (idx >= dev->submit_bo_list_size) {
      uint32_t new_len = idx + 64;
      struct drm_msm_gem_submit_bo *new_ptr = (struct drm_msm_gem_submit_bo *)
         vk_realloc(&dev->vk.alloc, dev->submit_bo_list, new_len * sizeof(*dev->submit_bo_list),
                    8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!new_ptr) {
         dev->submit_bo_count--;
         mtx_unlock(&dev->bo_mutex);
         vdrm_bo_close(dev->vdev->vdrm, gem_handle);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      dev->submit_bo_list = new_ptr;
      dev->submit_bo_list_size = new_len;
   }

   bool implicit_sync = flags & TU_BO_ALLOC_IMPLICIT_SYNC;
   bool dump = flags & TU_BO_ALLOC_ALLOW_DUMP;
   dev->submit_bo_list[idx] = (struct drm_msm_gem_submit_bo) {
      .flags = MSM_SUBMIT_BO_READ | MSM_SUBMIT_BO_WRITE |
               COND(dump, MSM_SUBMIT_BO_DUMP) |
               COND(!implicit_sync, MSM_SUBMIT_BO_NO_IMPLICIT),
      .handle = bo->res_id,
      .presumed = iova,
   };

   if (implicit_sync)
      dev->implicit_sync_bo_count++;

   *bo = (struct tu_bo) {
      .gem_handle = gem_handle,
      .res_id = bo->res_id,
      .size = size,
      .iova = iova,
      .name = name,
      .refcnt = 1,
      .submit_bo_list_idx = idx,
      .implicit_sync = implicit_sync,
      .base = base,
   };

   mtx_unlock(&dev->bo_mutex);

   tu_dump_bo_init(dev, bo);

   return VK_SUCCESS;
}

/**
 * Sets the name in the kernel so that the contents of /debug/dri/0/gem are more
 * useful.
 *
 * We skip this on release builds (when we're also not doing BO debugging) to
 * reduce overhead.
 */
static void
tu_bo_set_kernel_name(struct tu_device *dev, struct tu_bo *bo, const char *name)
{
   MESA_TRACE_FUNC();
   bool kernel_bo_names = dev->bo_sizes != NULL;
#if MESA_DEBUG
   kernel_bo_names = true;
#endif
   if (!kernel_bo_names)
      return;

   size_t sz = strlen(name);

   unsigned req_len = sizeof(struct msm_ccmd_gem_set_name_req) + align(sz, 4);

   uint8_t buf[req_len];
   struct msm_ccmd_gem_set_name_req *req = (struct msm_ccmd_gem_set_name_req *)buf;

   req->hdr = MSM_CCMD(GEM_SET_NAME, req_len);
   req->res_id = bo->res_id;
   req->len = sz;

   memcpy(req->payload, name, sz);

   vdrm_send_req(dev->vdev->vdrm, &req->hdr, false);
}

/* Common body for virtio_bo_init and (on Windows) virtio_bo_init_shared.
 *
 * hRTResource selects where the platform allocation comes from: NULL means this
 * driver's own device (the ordinary path), non-NULL routes it through the D3D
 * UMD's runtime allocator so the result can become a D3D11 shared resource.
 * Everything else -- GEM_NEW flags, blob_id, IOVA reservation, size rounding --
 * must stay identical, because the host requires GEM_NEW and the later
 * RESOURCE_CREATE_BLOB to agree exactly on blob_id and size. */
static VkResult
virtio_bo_init_common(struct tu_device *dev,
                      struct vk_object_base *base,
                      struct tu_bo **out_bo,
                      uint64_t size,
                      uint64_t client_iova,
                      VkMemoryPropertyFlags mem_property,
                      enum tu_bo_alloc_flags flags,
                      struct tu_sparse_vma *lazy_vma,
                      const char *name,
                      void *hRTResource)
{
   MESA_TRACE_FUNC();
   struct tu_virtio_device *vdev = dev->vdev;
#ifdef _WIN32
   /* The WDDM guest allocator uses 64 KiB alignment. Round before reserving
    * the IOVA as well as before GEM_NEW/CREATE_BLOB, so adjacent BOs cannot
    * overlap when imported by hosts with larger pages. */
   if (vdev->vdrm->supports_guest_alloc) {
      if (size > UINT64_MAX - 65535)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      size = ALIGN_POT(size, 65536);
   }
#endif
   struct msm_ccmd_gem_new_req req = {
         .hdr = MSM_CCMD(GEM_NEW, sizeof(req)),
         .size = size,
   };
   VkResult result = VK_SUCCESS;
   uint32_t res_id;
   struct tu_bo *bo;

   if (mem_property & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) {
      if (mem_property & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {
         req.flags |= MSM_BO_CACHED_COHERENT;
      } else {
         req.flags |= MSM_BO_CACHED;
      }
   } else {
      req.flags |= MSM_BO_WC;
   }

   /* DroidVM guest-alloc: tell the host not to allocate anything for this BO -- the guest
    * kernel backs the blob from its own pool and the VMM hands the host a dma-buf over those
    * pages. vdrm sets the blob flags that make that happen; this is the half the host's msm
    * command stream needs, because GEM_NEW arrives before the pages do and would otherwise
    * allocate a second, unused backing. */
   if (vdev->vdrm->supports_guest_alloc) {
      req.flags |= MSM_BO_GUEST_ALLOC;
#ifdef _WIN32
      /* KMD's contiguous allocation and its userspace alias are cacheable.
       * Import it IO-coherently; do not claim a WC CPU mapping to KGSL. */
      req.flags &= ~(MSM_BO_CACHED | MSM_BO_WC | MSM_BO_UNCACHED);
      req.flags |= MSM_BO_CACHED_COHERENT;
#endif
   }

   uint32_t blob_flags = 0;
   if (mem_property & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      blob_flags |= VIRTGPU_BLOB_FLAG_USE_MAPPABLE;
   }

   if (!(mem_property & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)) {
      blob_flags |= VIRTGPU_BLOB_FLAG_USE_SHAREABLE;
      if (vdev->vdrm->supports_cross_device)
         blob_flags |= VIRTGPU_BLOB_FLAG_USE_CROSS_DEVICE;
   }

   if (flags & TU_BO_ALLOC_GPU_READ_ONLY)
      req.flags |= MSM_BO_GPU_READONLY;

   assert(!(flags & TU_BO_ALLOC_DMABUF));

   if (lazy_vma) {
      req.iova = lazy_vma->msm.iova;
   } else {
      mtx_lock(&dev->vma_mutex);
      result = virtio_allocate_userspace_iova_locked(dev, 0, size, client_iova,
                                                     flags, &req.iova);
      mtx_unlock(&dev->vma_mutex);
      if (result != VK_SUCCESS && hRTResource)
         mesa_logw("virtio_bo_init_common: iova alloc failed size=%" PRIu64, size);
   }

   if (result != VK_SUCCESS)
      return result;

   /* tunneled cmds are processed separately on host side,
    * before the renderer->get_blob() callback.. the blob_id
    * is used to link the created bo to the get_blob() call
    */
   req.blob_id = p_atomic_inc_return(&vdev->next_blob_id);;

   uint32_t handle;
#ifdef _WIN32
   if (hRTResource) {
      /* The UMD mints hKMResource inside its alloc hook and keeps it; we only
       * need to know the allocation succeeded. */
      handle = vdrm_bo_create_shared(vdev->vdrm, size, blob_flags, req.blob_id,
                                     0, &req.hdr, hRTResource, NULL);
   } else
#else
   (void)hRTResource;
#endif
   {
      handle =
         vdrm_bo_create(vdev->vdrm, size, blob_flags, req.blob_id, 0, &req.hdr);
   }

   if (!handle) {
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      goto fail;
   }

   res_id = vdrm_handle_to_res_id(vdev->vdrm, handle);
   bo = tu_device_lookup_bo(dev, res_id);
   assert(bo && bo->gem_handle == 0);

   bo->res_id = res_id;

   result = tu_bo_init(dev, base, bo, handle, size, req.iova, flags, name);
   if (result != VK_SUCCESS) {
      memset(bo, 0, sizeof(*bo));
      goto fail;
   }

   if (lazy_vma) {
      lazy_vma->msm.backs_lazy_bo = true;
      bo->lazy = true;
   }

   *out_bo = bo;
   if (lazy_vma)
      lazy_vma->msm.backs_lazy_bo = true;

   /* We don't use bo->name here because for the !TU_DEBUG=bo case bo->name is NULL. */
   tu_bo_set_kernel_name(dev, bo, name);

   if ((mem_property & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) &&
       !(mem_property & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
      tu_bo_map(dev, bo, NULL);

      /* Cached non-coherent memory may already have dirty cache lines,
       * we should clean the cache lines before GPU got the chance to
       * write into this memory.
       *
       * MSM already does this automatically for uncached (MSM_BO_WC) memory.
       */
      tu_bo_sync_cache(dev, bo, 0, VK_WHOLE_SIZE, TU_MEM_SYNC_CACHE_TO_GPU);
   }

   return VK_SUCCESS;

fail:
   if (!lazy_vma) {
      mtx_lock(&dev->vma_mutex);
      util_vma_heap_free(&dev->vma, req.iova, size);
      mtx_unlock(&dev->vma_mutex);
   }
   return result;
}

static VkResult
virtio_bo_init(struct tu_device *dev,
               struct vk_object_base *base,
               struct tu_bo **out_bo,
               uint64_t size,
               uint64_t client_iova,
               VkMemoryPropertyFlags mem_property,
               enum tu_bo_alloc_flags flags,
               struct tu_sparse_vma *lazy_vma,
               const char *name)
{
   return virtio_bo_init_common(dev, base, out_bo, size, client_iova,
                                mem_property, flags, lazy_vma, name, NULL);
}

#ifdef _WIN32
static VkResult
virtio_bo_init_shared(struct tu_device *dev,
                      struct vk_object_base *base,
                      struct tu_bo **out_bo,
                      uint64_t size,
                      uint64_t client_iova,
                      VkMemoryPropertyFlags mem_property,
                      enum tu_bo_alloc_flags flags,
                      const char *name,
                      void *hRTResource)
{
   /* A shared BO cannot back a sparse VMA: lazy_vma exists so several VMAs can
    * share one BO, which has no meaning for a resource the runtime owns. */
   assert(hRTResource);
   return virtio_bo_init_common(dev, base, out_bo, size, client_iova,
                                mem_property, flags, NULL, name, hRTResource);
}
#endif

static VkResult
virtio_bo_init_dmabuf(struct tu_device *dev,
                   struct tu_bo **out_bo,
                   uint64_t size,
                   enum tu_bo_alloc_flags flags,
                   int prime_fd)
{
   MESA_TRACE_FUNC();
   struct vdrm_device *vdrm = dev->vdev->vdrm;
   VkResult result;
   struct tu_bo* bo = NULL;
   flags = (enum tu_bo_alloc_flags)(flags | TU_BO_ALLOC_DMABUF);

   /* lseek() to get the real size */
   off_t real_size = lseek(prime_fd, 0, SEEK_END);
   lseek(prime_fd, 0, SEEK_SET);
   if (real_size < 0 || (uint64_t) real_size < size)
      return vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);

   /* iova allocation needs to consider the object's *real* size: */
   size = real_size;

   /* Importing the same dmabuf several times would yield the same
    * gem_handle. Thus there could be a race when destroying
    * BO and importing the same dmabuf from different threads.
    * We must not permit the creation of dmabuf BO and its release
    * to happen in parallel.
    */
   u_rwlock_wrlock(&dev->dma_bo_lock);

   uint32_t handle, res_id;
   uint64_t iova;

   handle = vdrm_dmabuf_to_handle(vdrm, prime_fd);
   if (!handle) {
      result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      goto out_unlock;
   }

   res_id = vdrm_handle_to_res_id(vdrm, handle);
   if (!res_id) {
      vdrm_bo_close(vdrm, handle);
      result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      goto out_unlock;
   }

   bo = tu_device_lookup_bo(dev, res_id);

   if (bo->refcnt != 0) {
      p_atomic_inc(&bo->refcnt);
      assert(bo->res_id == res_id);
      *out_bo = bo;
      result = VK_SUCCESS;
      goto out_unlock;
   }

   bo->res_id = res_id;

   mtx_lock(&dev->vma_mutex);
   result = virtio_allocate_userspace_iova_locked(dev, handle, size, 0, flags,
                                                  &iova);
   mtx_unlock(&dev->vma_mutex);
   if (result != VK_SUCCESS) {
      vdrm_bo_close(vdrm, handle);
      goto out_unlock;
   }

   result =
      tu_bo_init(dev, NULL, bo, handle, size, iova, flags, "dmabuf");
   if (result != VK_SUCCESS) {
      mtx_lock(&dev->vma_mutex);
      util_vma_heap_free(&dev->vma, iova, size);
      mtx_unlock(&dev->vma_mutex);
      memset(bo, 0, sizeof(*bo));
   } else {
      *out_bo = bo;
      set_iova(dev, bo->res_id, iova);
   }

out_unlock:
   u_rwlock_wrunlock(&dev->dma_bo_lock);
   return result;
}

#ifdef _WIN32
/* Import a BO that some other component already allocated -- on Windows the UMD
 * creates the shared WDDM allocation and another process may have opened it via
 * OpenResource2.  Mirrors virtio_bo_init_dmabuf: the only difference is how we
 * get from a platform handle to a vdrm handle, because Windows has no PRIME fd.
 *
 * The refcount keys on res_id, so two imports of the same backing inside one
 * device collapse onto one tu_bo, exactly as two imports of one dmabuf do.
 * Across devices/processes the iova is allocated per device, which is required:
 * the host hands each context a disjoint VA slice (kgsl_iova_in_slice), so the
 * same backing legitimately sits at different GPU addresses in each context.
 */
static VkResult
virtio_bo_init_platform_alloc(struct tu_device *dev,
                              struct tu_bo **out_bo,
                              uint64_t size,
                              enum tu_bo_alloc_flags flags,
                              uint32_t platform_handle)
{
   MESA_TRACE_FUNC();
   struct vdrm_device *vdrm = dev->vdev->vdrm;
   VkResult result;
   struct tu_bo *bo = NULL;
   uint64_t real_size = 0;

   /* Same serialization as the dmabuf path: importing the same backing from two
    * threads must not race with a concurrent release of that tu_bo. */
   u_rwlock_wrlock(&dev->dma_bo_lock);

   uint32_t handle, res_id;
   uint64_t iova;

   handle = vdrm_alloc_to_handle(vdrm, platform_handle, &real_size);
   if (!handle) {
      /* Either this backend has no import entry point, or the handle is not a
       * blob we can adopt.  Both are handle-level failures from the caller's
       * point of view; tu_bo_init_platform_alloc already reported the
       * unsupported-backend case before reaching here. */
      result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      goto out_unlock;
   }

   /* Trust the owner's size over the caller's estimate (it may have rounded up
    * to the 64 KiB guest-alloc granule); a short iova range would only surface
    * later as a GPU fault. Never accept less than the caller needs. */
   if (real_size < size) {
      vdrm_bo_close(vdrm, handle);
      result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      goto out_unlock;
   }
   size = real_size;

   res_id = vdrm_handle_to_res_id(vdrm, handle);
   if (!res_id) {
      vdrm_bo_close(vdrm, handle);
      result = vk_error(dev, VK_ERROR_INVALID_EXTERNAL_HANDLE);
      goto out_unlock;
   }

   bo = tu_device_lookup_bo(dev, res_id);

   if (bo->refcnt != 0) {
      /* Already imported in this device; drop the duplicate vdrm handle so we
       * do not leak a table slot (the allocation itself is not ours). */
      p_atomic_inc(&bo->refcnt);
      assert(bo->res_id == res_id);
      vdrm_bo_close(vdrm, handle);
      *out_bo = bo;
      result = VK_SUCCESS;
      goto out_unlock;
   }

   bo->res_id = res_id;

   mtx_lock(&dev->vma_mutex);
   result = virtio_allocate_userspace_iova_locked(dev, handle, size, 0, flags,
                                                  &iova);
   mtx_unlock(&dev->vma_mutex);
   if (result != VK_SUCCESS) {
      vdrm_bo_close(vdrm, handle);
      goto out_unlock;
   }

   result = tu_bo_init(dev, NULL, bo, handle, size, iova, flags, "imported");
   if (result != VK_SUCCESS) {
      mtx_lock(&dev->vma_mutex);
      util_vma_heap_free(&dev->vma, iova, size);
      mtx_unlock(&dev->vma_mutex);
      memset(bo, 0, sizeof(*bo));
   } else {
      *out_bo = bo;
      set_iova(dev, bo->res_id, iova);
   }

out_unlock:
   u_rwlock_wrunlock(&dev->dma_bo_lock);
   return result;
}
#endif /* _WIN32 */

static int
virtio_bo_export_dmabuf(struct tu_device *dev, struct tu_bo *bo)
{
   MESA_TRACE_FUNC();
   return vdrm_bo_export_dmabuf(dev->vdev->vdrm, bo->gem_handle);
}

static VkResult
virtio_bo_map(struct tu_device *dev, struct tu_bo *bo, void *placed_addr)
{
   MESA_TRACE_FUNC();
   bo->map = vdrm_bo_map(dev->vdev->vdrm, bo->gem_handle, bo->size, placed_addr);
   if (bo->map == MAP_FAILED)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);

   return VK_SUCCESS;
}

static void
virtio_bo_allow_dump(struct tu_device *dev, struct tu_bo *bo)
{
   mtx_lock(&dev->bo_mutex);
   dev->submit_bo_list[bo->submit_bo_list_idx].flags |= MSM_SUBMIT_BO_DUMP;
   mtx_unlock(&dev->bo_mutex);
}

static void
virtio_bo_finish(struct tu_device *dev, struct tu_bo *bo)
{
   assert(bo->gem_handle);

   u_rwlock_rdlock(&dev->dma_bo_lock);

   if (!p_atomic_dec_zero(&bo->refcnt)) {
      u_rwlock_rdunlock(&dev->dma_bo_lock);
      return;
   }

   tu_debug_bos_del(dev, bo);
   tu_dump_bo_del(dev, bo);

   if (bo->map) {
#if defined(_WIN32)
      /* TODO(块 4): vdrm_wddm 的 UNMAP（VIRTIO_WDDM_BLOB_MAP_FLAGS_UNMAP）。 */
#else
      munmap(bo->map, bo->size);
#endif
   }

   tu_bo_list_del(dev, bo);

   assert(dev->physical_device->has_set_iova);
   tu_bo_make_zombie(dev, bo);

#if defined(_WIN32)
   /* VidMm owns the allocation's commitment until vdrm_bo_close(). Waiting
    * for the next BO allocation to reap zombies leaves freed memory charged
    * indefinitely in an idle process. Reuse the nonblocking fence-checked
    * retirement path; unrelated in-flight work must still keep its BOs alive.
    */
   /* Device setup/teardown can release BOs before the fence mapping exists
    * or after global_bo itself has become a zombie. Leave those to device
    * destruction, since tu_wait_fence() reads that mapping. */
   if (dev->global_bo && dev->global_bo->gem_handle && dev->global_bo_map) {
      mtx_lock(&dev->vma_mutex);
      tu_free_zombie_vma_locked(dev, false);
      mtx_unlock(&dev->vma_mutex);
   }
#endif

   u_rwlock_rdunlock(&dev->dma_bo_lock);
}

static VkResult
virtio_sparse_vma_init(struct tu_device *dev,
                       struct vk_object_base *base,
                       struct tu_sparse_vma *out_vma,
                       uint64_t *out_iova,
                       enum tu_sparse_vma_flags flags,
                       uint64_t size, uint64_t client_iova)
{
   VkResult result;
   enum tu_bo_alloc_flags bo_flags =
      (flags & TU_SPARSE_VMA_REPLAYABLE) ? TU_BO_ALLOC_REPLAYABLE :
      (enum tu_bo_alloc_flags)0;

   out_vma->msm.size = size;

   mtx_lock(&dev->vma_mutex);
   result = virtio_allocate_userspace_iova_locked(dev, 0, size, client_iova,
                                                  bo_flags, &out_vma->msm.iova);
   mtx_unlock(&dev->vma_mutex);

   if (result != VK_SUCCESS)
      return result;

   assert(!(flags & TU_SPARSE_VMA_MAP_ZERO));

   *out_iova = out_vma->msm.iova;

   return result;
}

static void
virtio_sparse_vma_finish(struct tu_device *dev,
                         struct tu_sparse_vma *vma)
{
   /* A lazy backing BO transfers this reservation to the zombie VMA path,
    * which releases it only after the last GPU fence has retired. */
   if (!vma->msm.backs_lazy_bo) {
      mtx_lock(&dev->vma_mutex);
      util_vma_heap_free(&dev->vma, vma->msm.iova, vma->msm.size);
      mtx_unlock(&dev->vma_mutex);
   }
}

static VkResult
setup_fence_cmds(struct tu_device *dev)
{
   MESA_TRACE_FUNC();
   struct tu_virtio_device *vdev = dev->vdev;
   VkResult result;

   result = tu_bo_init_new(dev, NULL, &vdev->fence_cmds_mem,
                           sizeof(*vdev->fence_cmds), (enum tu_bo_alloc_flags)
                              (TU_BO_ALLOC_ALLOW_DUMP | TU_BO_ALLOC_GPU_READ_ONLY),
                           "fence_cmds");
   if (result != VK_SUCCESS)
      return result;

   result = tu_bo_map(dev, vdev->fence_cmds_mem, NULL);
   if (result != VK_SUCCESS)
      return result;

   vdev->fence_cmds = (struct tu_userspace_fence_cmds *)vdev->fence_cmds_mem->map;

   uint64_t fence_iova = dev->global_bo->iova + gb_offset(userspace_fence);
   for (int i = 0; i < ARRAY_SIZE(vdev->fence_cmds->cmds); i++) {
      struct tu_userspace_fence_cmd *c = &vdev->fence_cmds->cmds[i];

      memset(c, 0, sizeof(*c));

      if (fd_dev_gen(&dev->physical_device->dev_id) >= A7XX) {
         c->pkt[0] = pm4_pkt7_hdr((uint8_t)CP_EVENT_WRITE7, 4);
         c->pkt[1] = CP_EVENT_WRITE7_0(.event = CACHE_FLUSH_TS,
                           .write_src = EV_WRITE_USER_32B,
                           .write_dst = EV_DST_RAM,
                           .write_enabled = true).value;
      } else {
         c->pkt[0] = pm4_pkt7_hdr((uint8_t)CP_EVENT_WRITE, 4);
         c->pkt[1] = CP_EVENT_WRITE_0_EVENT(CACHE_FLUSH_TS);
      }
      c->pkt[2] = fence_iova;
      c->pkt[3] = fence_iova >> 32;
   }

   /* Guest-pool pages retain their contents across BO lifetimes.  Start a
    * fresh fence generation and publish both CPU-written buffers before the
    * poll-first path can use them.
    */
   dev->global_bo_map->userspace_fence = 0;
   if (vdev->vdrm->supports_guest_alloc) {
      virtio_sync_guest_bo_cache(dev, dev->global_bo, gb_offset(userspace_fence),
                       sizeof(dev->global_bo_map->userspace_fence),
                       TU_MEM_SYNC_CACHE_TO_GPU);
      virtio_sync_guest_bo_cache(dev, vdev->fence_cmds_mem, 0, VK_WHOLE_SIZE,
                       TU_MEM_SYNC_CACHE_TO_GPU);
   }

   return result;
}

static VkResult
virtio_queue_submit(struct tu_queue *queue, void *_submit,
                    struct vk_sync_wait *waits, uint32_t wait_count,
                    struct vk_sync_signal *signals, uint32_t signal_count,
                    struct tu_u_trace_submission_data *u_trace_submission_data)
{
   MESA_TRACE_FUNC();
   VkResult result = VK_SUCCESS;
   int ret;
   struct tu_msm_queue_submit *submit =
      (struct tu_msm_queue_submit *)_submit;
   struct tu_virtio_device *vdev = queue->device->vdev;
   struct drm_virtgpu_execbuffer_syncobj *in_syncobjs, *out_syncobjs;
   uint64_t gpu_offset = 0;
   int ring_idx = queue->priority + 1;
   uint32_t num_out_syncobjs = 0;
   uint32_t num_in_syncobjs = wait_count;
   struct vdrm_execbuf_params params;

   if (tu_empty_submit_can_skip(queue, submit, waits, wait_count,
                                signals, signal_count,
                                u_trace_submission_data))
      return tu_empty_submit_inherit(queue, signals, signal_count);

   /* Tier 2: empty submit with arbitrary waits (WSI acquire, cross-queue).
    * Resolve with copy_payloads if available, falling back to EXECBUFFER. */
   if (submit->commands.size == 0 && submit->binds.size == 0 &&
       !u_trace_submission_data) {
      VkResult copy_result = tu_empty_submit_copy_payloads(queue, waits, wait_count,
                                                           signals, signal_count);
      if (copy_result == VK_SUCCESS)
         return VK_SUCCESS;
      /* FEATURE_NOT_PRESENT or disabled → fall through to real submit */
   }

#ifdef _WIN32
   /* Windows 没有 drm syncobj，execbuf 带不了 in/out syncobj（vdrm_wddm 的
    * execbuf_locked 见到非 0 计数直接 -ENOSYS）。等待改成提交前的 CPU 等待：
    * 保守但正确——等到之后再入 ring，host 按序执行，语义强于 GPU 侧等待。
    * 放在拿 bo_mutex 之前，别拿着锁睡。信号见下面提交后的 seqno 记录。 */
   if (wait_count) {
      result = vk_sync_wait_many(&queue->device->vk, wait_count, waits,
                                 (enum vk_sync_wait_flags)0,
                                 OS_TIMEOUT_INFINITE);
      if (result != VK_SUCCESS)
         return result;
   }
   num_in_syncobjs = 0;
#endif

#if HAVE_PERFETTO
   struct tu_perfetto_clocks clocks;
   uint64_t start_ts = tu_perfetto_begin_submit();
#endif

   /* It would be nice to not need to defer this, but virtio_device_init()
    * happens before the device is initialized enough to allocate normal
    * GEM buffers
    */
   if (!vdev->fence_cmds) {
      VkResult result = setup_fence_cmds(queue->device);
      if (result != VK_SUCCESS)
         return result;
   }

   /* Add the userspace fence cmd: */
   struct tu_userspace_fence_cmds *fcmds = vdev->fence_cmds;
   if (queue->fence <= 0)
      queue->fence = 0;
   uint32_t fence = ++queue->fence;
   int idx = fence % ARRAY_SIZE(fcmds->cmds);
   fcmds->cmds[idx].fence = fence;
   if (vdev->vdrm->supports_guest_alloc) {
      virtio_sync_guest_bo_cache(queue->device, vdev->fence_cmds_mem,
                       (uintptr_t)&fcmds->cmds[idx] - (uintptr_t)fcmds,
                       sizeof(fcmds->cmds[idx]), TU_MEM_SYNC_CACHE_TO_GPU);
   }
   struct tu_cs_entry fence_cs = {
      .bo = vdev->fence_cmds_mem,
      .size = 5 * 4,
      .offset = ((intptr_t)&fcmds->cmds[idx]) - (intptr_t)fcmds,
   };
   msm_submit_add_entries(queue->device, _submit, &fence_cs, 1);

   uint32_t entry_count =
      util_dynarray_num_elements(&submit->commands, struct drm_msm_gem_submit_cmd);
   unsigned nr_bos = entry_count ? queue->device->submit_bo_count : 0;
   unsigned bos_len = nr_bos * sizeof(struct drm_msm_gem_submit_bo);
   unsigned cmd_len = entry_count * sizeof(struct drm_msm_gem_submit_cmd);
   unsigned req_len = sizeof(struct msm_ccmd_gem_submit_req) + bos_len + cmd_len;
   struct msm_ccmd_gem_submit_req *req;
   uint32_t flags = MSM_PIPE_3D0;

   /* Allocate without wait timeline semaphores */
   in_syncobjs = (struct drm_virtgpu_execbuffer_syncobj *) vk_zalloc(
      &queue->device->vk.alloc,
      wait_count * sizeof(*in_syncobjs), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);

   if (in_syncobjs == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_in_syncobjs;
   }

   /* Allocate with signal timeline semaphores considered, plus one slot for
    * the last-submit tracking syncobj (empty-submit fast path). */
   out_syncobjs = (struct drm_virtgpu_execbuffer_syncobj *) vk_zalloc(
      &queue->device->vk.alloc,
      (signal_count + 1) * sizeof(*out_syncobjs), 8,
      VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);

   if (out_syncobjs == NULL) {
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_out_syncobjs;
   }

   for (uint32_t i = 0; i < num_in_syncobjs; i++) {
      struct vk_sync *sync = waits[i].sync;

      in_syncobjs[i] = (struct drm_virtgpu_execbuffer_syncobj) {
         .handle = vk_sync_as_drm_syncobj(sync)->syncobj,
         .flags = 0,
         .point = waits[i].wait_value,
      };
   }

#ifdef _WIN32
   /* 信号也不走 syncobj：见提交之后的 seqno 记录。 */
   num_out_syncobjs = 0;
#else
   num_out_syncobjs = signal_count;
   for (uint32_t i = 0; i < signal_count; i++) {
      struct vk_sync *sync = signals[i].sync;

      out_syncobjs[i] = (struct drm_virtgpu_execbuffer_syncobj) {
         .handle = vk_sync_as_drm_syncobj(sync)->syncobj,
         .flags = 0,
         .point = signals[i].signal_value,
      };
   }

   /* Park this submit's fence in the tracking syncobj so a later empty
    * submit can inherit it without a host round-trip.  Gated behind the same
    * env as the skip so TU_NO_EMPTY_SUBMIT=1 is a complete revert for A/B. */
   if (vdev->last_submit_syncobj && !tu_empty_submit_disabled()) {
      out_syncobjs[num_out_syncobjs++] = (struct drm_virtgpu_execbuffer_syncobj) {
         .handle = vdev->last_submit_syncobj,
         .flags = 0,
         .point = 0,
      };
   }
#endif /* !_WIN32 */

   if (num_in_syncobjs)
      flags |= MSM_SUBMIT_SYNCOBJ_IN;

   if (num_out_syncobjs)
      flags |= MSM_SUBMIT_SYNCOBJ_OUT;

   mtx_lock(&queue->device->bo_mutex);

   if (queue->device->implicit_sync_bo_count == 0)
      flags |= MSM_SUBMIT_NO_IMPLICIT;

   /* drm_msm_gem_submit_cmd requires index of bo which could change at any
    * time when bo_mutex is not locked. So we update the index here under the
    * lock.
    */
   util_dynarray_foreach (&submit->commands, struct drm_msm_gem_submit_cmd,
                          cmd) {
      unsigned i = cmd -
         util_dynarray_element(&submit->commands,
                               struct drm_msm_gem_submit_cmd, 0);
      struct tu_bo **bo = util_dynarray_element(&submit->command_bos,
                                                struct tu_bo *, i);
      cmd->submit_idx = (*bo)->submit_bo_list_idx;
   }

   req = (struct msm_ccmd_gem_submit_req *)vk_alloc(
         &queue->device->vk.alloc, req_len, 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);

   if (!req) {
      mtx_unlock(&queue->device->bo_mutex);
      result = vk_error(queue, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail_alloc_req;
   }

   req->hdr      = MSM_CCMD(GEM_SUBMIT, req_len);
   req->flags    = flags;
   req->queue_id = queue->msm_queue_id;
   req->nr_bos   = nr_bos;
   req->nr_cmds  = entry_count;

   /* Use same kernel fence and userspace fence seqno to avoid having
    * to track both:
    */
   req->fence    = queue->fence;

   memcpy(req->payload, queue->device->submit_bo_list, bos_len);
   memcpy(req->payload + bos_len, submit->commands.data, cmd_len);

   params = (struct vdrm_execbuf_params) {
      .ring_idx = ring_idx,
      .req = &req->hdr,
      .in_syncobjs = in_syncobjs,
      .out_syncobjs = out_syncobjs,
      .num_in_syncobjs = num_in_syncobjs,
      .num_out_syncobjs = num_out_syncobjs,
   };

#ifdef _WIN32
   if (debug_get_option_tu_win_sync_debug())
      fprintf(stderr, "tu-win32-sync: submit queue=%u fence=%u bos=%u "
              "cmds=%u waits=%u signals=%u bytes=%u\n",
              req->queue_id, fence, nr_bos, entry_count,
              wait_count, signal_count, req_len);
#endif

   ret = vdrm_execbuf(vdev->vdrm, &params);

#ifdef _WIN32
   if (debug_get_option_tu_win_sync_debug())
      fprintf(stderr, "tu-win32-sync: enqueued ret=%d req=%u host_seq=%u "
              "gpu=%u async_error=%u\n", ret, req->hdr.seqno,
              vdev->shmem->base.seqno,
              queue->device->global_bo_map->userspace_fence,
              vdev->shmem->async_error);
#endif

   mtx_unlock(&queue->device->bo_mutex);

   if (ret) {
      result = vk_device_set_lost(&queue->device->vk, "submit failed: %m");
      goto fail_submit;
   }

#ifdef _WIN32
   /* Windows 上这是**唯一**的信号机制（没有 syncobj）：把本次提交的 seqno 记到
    * 每个 signal sync 上，等待方拿它跟 userspace_fence 比（tu_win_sync_type /
    * tu_virtio_sync_wait_many）。signal sync 一定是 syncobj_type 或它的 poll
    * 包装，两者载体同为 struct tu_virtio_sync，所以直接转型——与 Linux 那边对
    * vk_sync_as_drm_syncobj() 的无条件转型同理。
    * 多队列时 userspace_fence 有多个写者，seqno 不再全序 —— 与 Linux 的
    * tu_virtio_single_queue 限制相同，Windows 上没有退路可走，先记下来。 */
   for (uint32_t i = 0; i < signal_count; i++) {
      struct tu_virtio_sync *s = to_tu_virtio_sync(signals[i].sync);

      s->submit_seqno = fence;
      s->signaled = false;
      p_atomic_set(&s->owner, queue);
   }
#else
   /* Record the seqno on poll-type signal syncs so CPU waits can take the
    * userspace-fence fast path.  Only valid while this queue is the sole
    * writer of global_bo->userspace_fence. */
   if (tu_virtio_single_queue(queue->device)) {
      const struct vk_sync_type *poll_type =
         &queue->device->physical_device->poll_sync_type;

      for (uint32_t i = 0; i < signal_count; i++) {
         if (signals[i].sync->type == poll_type) {
            struct tu_virtio_sync *s = to_tu_virtio_sync(signals[i].sync);
            s->submit_seqno = fence;
            p_atomic_set(&s->owner, queue);
         }
      }
   }
#endif

#if HAVE_PERFETTO
   clocks = tu_perfetto_end_submit(queue, queue->device->submit_count,
                                   start_ts, NULL);
   gpu_offset = clocks.gpu_ts_offset;
#endif

   if (u_trace_submission_data) {
      u_trace_submission_data->gpu_ts_offset = gpu_offset;
   }

fail_submit:
   vk_free(&queue->device->vk.alloc, req);
fail_alloc_req:
   vk_free(&queue->device->vk.alloc, out_syncobjs);
fail_out_syncobjs:
   vk_free(&queue->device->vk.alloc, in_syncobjs);
fail_in_syncobjs:
   return result;
}

static const struct tu_knl virtio_knl_funcs = {
      .name = "virtgpu",

      .device_init = virtio_device_init,
      .device_finish = virtio_device_finish,
      .device_get_gpu_timestamp = virtio_device_get_gpu_timestamp,
      .device_get_suspend_count = virtio_device_get_suspend_count,
      .device_check_status = virtio_device_check_status,
      .submitqueue_new = virtio_submitqueue_new,
      .submitqueue_close = virtio_submitqueue_close,
      .bo_init = virtio_bo_init,
      .bo_init_dmabuf = virtio_bo_init_dmabuf,
#ifdef _WIN32
      .bo_init_platform_alloc = virtio_bo_init_platform_alloc,
      .bo_init_shared = virtio_bo_init_shared,
#endif
      .bo_export_dmabuf = virtio_bo_export_dmabuf,
      .bo_map = virtio_bo_map,
      .bo_allow_dump = virtio_bo_allow_dump,
      .bo_finish = virtio_bo_finish,
      .submit_create = msm_submit_create,
      .submit_finish = msm_submit_finish,
      .submit_add_entries = msm_submit_add_entries,
      .queue_submit = virtio_queue_submit,
      .queue_wait_fence = virtio_queue_wait_fence,
      .sparse_vma_init = virtio_sparse_vma_init,
      .sparse_vma_finish = virtio_sparse_vma_finish,
};

VkResult
tu_knl_drm_virtio_load(struct tu_instance *instance,
                       int fd, struct _drmVersion *version,
                       struct tu_physical_device **out)
{
   struct virgl_renderer_capset_drm caps;
   struct vdrm_device *vdrm;
   VkResult result = VK_SUCCESS;
   uint64_t val;

   /* Debug option to force fallback to venus: */
   if (debug_get_bool_option("TU_NO_VIRTIO", false))
      return VK_ERROR_INCOMPATIBLE_DRIVER;

#if !defined(_WIN32)
   /* Note: in vtest case, where we don't open a device fd directly, we
    * can't do drm ioctls directly.  But we can assume that the server
    * side supports syncobjs.
    */
   if ((fd >= 0) && (drmGetCap(fd, DRM_CAP_SYNCOBJ, &val) || !val)) {
      return vk_startup_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                               "kernel driver for device %s does not support DRM_CAP_SYNC_OBJ",
                               version->name);
   }
#endif

   /* Try to connect. If this doesn't work, it's probably because we're running
    * in a non-Adreno VM. Unless startup debug info is specifically requested,
    * we should silently exit and let another Vulkan driver try probing instead.
    */
   vdrm = vdrm_device_connect(fd, VIRTGPU_DRM_CONTEXT_MSM);
   if (!vdrm) {
#if defined(_WIN32)
      /* vdrm_wddm 还没实现（块 2 起），Windows 上 probe 必然到此为止。 */
      return VK_ERROR_INCOMPATIBLE_DRIVER;
#else
      if (TU_DEBUG(STARTUP)) {
         return vk_startup_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                                  "could not get connect vdrm: %s", strerror(errno));
      } else {
         return VK_ERROR_INCOMPATIBLE_DRIVER;
      }
#endif
   }

   caps = vdrm->caps;

   bool has_preemption = virtio_has_preemption(vdrm);

   /* If virglrenderer is too old, we may need another round-trip to get this.
    */
   if (caps.u.msm.highest_bank_bit == 0)
      caps.u.msm.highest_bank_bit = tu_drm_get_highest_bank_bit(vdrm);

   uint32_t bank_swizzle_levels = tu_drm_get_ubwc_swizzle(vdrm);
   enum fdl_macrotile_mode macrotile_mode = tu_drm_get_macrotile_mode(vdrm);
   uint64_t uche_trap_base = tu_drm_get_uche_trap_base(vdrm);

#if defined(_WIN32)
   /* 没有 drm syncobj：用 userspace-fence 支撑的自足二元类型（块 5）。
    * poll 包装层照样套在它上面（载体同为 struct tu_virtio_sync）。 */
   (void) vdrm;
   (void) val;
   struct vk_sync_type syncobj_type = tu_win_sync_type;
#else
   /* If using vtest, vtest provides it's own sync provider.  Otherwise this
    * returns NULL and we fall back to using the syncobj ioctls directly:
    */
   struct util_sync_provider *sync = vdrm_vpipe_get_sync(vdrm);
   if (!sync)
      sync = util_sync_provider_drm(fd);
   struct vk_sync_type syncobj_type = vk_drm_syncobj_get_type_from_provider(sync);
   sync->finalize(sync);
#endif

   bool has_raytracing = tu_drm_get_raytracing(vdrm);

   /* DroidVM KGSL nctx: host hands each context a disjoint VA slice via
    * GET_PARAM (capset va is global; concurrent guest processes would
    * otherwise overlap iovas in the host's shared VBO). Query before close. */
   {
      uint64_t vs = 0, vz = 0;
      if (!tu_drm_get_param(vdrm, MSM_PARAM_VA_START, &vs) && vs &&
          !tu_drm_get_param(vdrm, MSM_PARAM_VA_SIZE, &vz) && vz) {
         caps.u.msm.va_start = vs;
         caps.u.msm.va_size = vz;
      }
   }
   vdrm_device_close(vdrm);

   mesa_logd("wire_format_version: %u", caps.wire_format_version);
   mesa_logd("version_major:       %u", caps.version_major);
   mesa_logd("version_minor:       %u", caps.version_minor);
   mesa_logd("version_patchlevel:  %u", caps.version_patchlevel);
   mesa_logd("has_cached_coherent: %u", caps.u.msm.has_cached_coherent);
   mesa_logd("va_start:            0x%0" PRIx64, caps.u.msm.va_start);
   mesa_logd("va_size:             0x%0" PRIx64, caps.u.msm.va_size);
   mesa_logd("gpu_id:              %u", caps.u.msm.gpu_id);
   mesa_logd("gmem_size:           %u", caps.u.msm.gmem_size);
   mesa_logd("gmem_base:           0x%0" PRIx64, caps.u.msm.gmem_base);
   mesa_logd("chip_id:             0x%0" PRIx64, caps.u.msm.chip_id);
   mesa_logd("max_freq:            %u", caps.u.msm.max_freq);
   mesa_logd("highest_bank_bit:    %u", caps.u.msm.highest_bank_bit);
   mesa_logd("ubwc_swizzle:        0x%" PRIx64, caps.u.msm.ubwc_swizzle);
   mesa_logd("macrotile_mode:      %" PRIu64, caps.u.msm.macrotile_mode);
   mesa_logd("has_raytracing:      %x", caps.u.msm.has_raytracing);
   mesa_logd("has_preemption:      %d", caps.u.msm.has_preemption);
   mesa_logd("uche_trap_base:      0x%" PRIx64, caps.u.msm.uche_trap_base);

   if (caps.wire_format_version != 2) {
      return vk_startup_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                               "Unsupported protocol version: %u",
                               caps.wire_format_version);
   }

   if ((caps.version_major != 1) || (caps.version_minor < 9)) {
      return vk_startup_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                               "unsupported version: %u.%u.%u",
                               caps.version_major,
                               caps.version_minor,
                               caps.version_patchlevel);
   }

   if (!caps.u.msm.va_size) {
      return vk_startup_errorf(instance, VK_ERROR_INCOMPATIBLE_DRIVER,
                               "No address space");
   }

   struct tu_physical_device *device = (struct tu_physical_device *)
      vk_zalloc(&instance->vk.alloc, sizeof(*device), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!device) {
      result = vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
      goto fail;
   }

   device->msm_major_version = caps.version_major;
   device->msm_minor_version = caps.version_minor;

   device->instance = instance;
   device->local_fd = fd;

   device->dev_id.gpu_id  = caps.u.msm.gpu_id;
   device->dev_id.chip_id = caps.u.msm.chip_id;
   device->gmem_size      = caps.u.msm.gmem_size;
   device->gmem_base      = caps.u.msm.gmem_base;
   device->va_start       = caps.u.msm.va_start;
   device->va_size        = caps.u.msm.va_size;
   device->ubwc_config.highest_bank_bit = caps.u.msm.highest_bank_bit;
   device->has_set_iova   = true;
   device->has_lazy_bos   = true;
   device->has_preemption = has_preemption;
   device->is_perf_cntr_selectable = true;
   device->uche_trap_base = uche_trap_base;

   device->ubwc_config.bank_swizzle_levels = bank_swizzle_levels;
   device->ubwc_config.macrotile_mode = macrotile_mode;

   device->gmem_size = debug_get_num_option("TU_GMEM", device->gmem_size);

   device->has_cached_coherent_memory = caps.u.msm.has_cached_coherent;

   device->submitqueue_priority_count = caps.u.msm.priorities;

   device->has_raytracing = has_raytracing;

   device->syncobj_type = syncobj_type;

   /* msm didn't expose DRM_CAP_SYNCOBJ_TIMELINE until kernel 6.15, so emulate timeline
    * semaphores if necessary.
    */
   if (!(device->syncobj_type.features & VK_SYNC_FEATURE_TIMELINE))
      device->timeline_type = vk_sync_timeline_get_type(&device->syncobj_type);

   device->poll_sync_type = tu_virtio_get_poll_sync_type(&device->syncobj_type);

   /* Fences and binary semaphores pick the poll-first wrapper; timeline
    * semaphores fall through to the plain syncobj type.  TU_NO_POLL_FIRST=1
    * reverts to the plain type for A/B comparison. */
   {
      unsigned st = 0;
      if (!debug_get_bool_option("TU_NO_POLL_FIRST", false) &&
          (device->poll_sync_type.features & VK_SYNC_FEATURE_CPU_WAIT))
         device->sync_types[st++] = &device->poll_sync_type;
      device->sync_types[st++] = &device->syncobj_type;
      device->sync_types[st++] = &device->timeline_type.sync;
      device->sync_types[st] = NULL;
   }

   /* DroidVM guest-alloc: when the VMM gave this guest a pool, every BO is backed out of it and
    * nothing else, so the pool is the heap -- report its size rather than a fraction of guest
    * RAM. Those are not the same number and the difference is not slack: a client that sizes its
    * suballocator against guest RAM (zink does) asks for hundreds of MiB the pool can never
    * satisfy, and gets ENOMEM out of the very first big allocation. Still capped by the address
    * space, same as the system-heap path. */
   if (vdrm_guest_pool_stats(fd, &device->guest_pool_size, NULL, NULL)) {
      device->heap.size = device->va_size ? MIN2(device->guest_pool_size, device->va_size)
                                          : device->guest_pool_size;
   } else {
      device->heap.size = tu_get_system_heap_size(device);
   }
   device->heap.used = 0u;
   device->heap.flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;

   instance->knl = &virtio_knl_funcs;

   *out = device;

   return VK_SUCCESS;

fail:
   vk_free(&instance->vk.alloc, device);
   return result;
}
