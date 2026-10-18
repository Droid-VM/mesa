/*
 * Copyright © 2023 Google, Inc.
 * SPDX-License-Identifier: MIT
 */

#include "util/u_math.h"
#include "util/perf/cpu_trace.h"
#include "c11/threads.h"

#include "vdrm.h"
#include "util/log.h"
#include "util/os_time.h"
#include "util/u_atomic.h"

#ifdef _WIN32
struct vdrm_device * vdrm_wddm_connect(int fd, uint32_t context_type);
#else
struct vdrm_device * vdrm_virtgpu_connect(int fd, uint32_t context_type);
struct vdrm_device * vdrm_vpipe_connect(uint32_t context_type);
#endif

struct vdrm_device *
vdrm_device_connect(int fd, uint32_t context_type)
{
   struct vdrm_device *vdev;

#ifdef _WIN32
   vdev = vdrm_wddm_connect(fd, context_type);
#else
   if (fd >= 0) {
      vdev = vdrm_virtgpu_connect(fd, context_type);
   } else {
      vdev = vdrm_vpipe_connect(context_type);
   }
#endif

   if (!vdev)
      return NULL;

   simple_mtx_init(&vdev->rsp_lock, mtx_plain);
   simple_mtx_init(&vdev->eb_lock, mtx_plain);

   return vdev;
}

void
vdrm_device_close(struct vdrm_device *vdev)
{
   vdev->funcs->close(vdev);
   free(vdev);
}

uint32_t
vdrm_bo_create(struct vdrm_device *vdev, size_t size, uint32_t blob_flags,
               uint64_t blob_id, uint32_t blob_hints,
               struct vdrm_ccmd_req *req)
{
   uint32_t handle;

   simple_mtx_lock(&vdev->eb_lock);

   /* flush any buffered cmds so they are seen by the host *prior* to
    * the cmds associated with bo creation.
    */
   vdev->funcs->flush_locked(vdev, NULL);

   req->seqno = ++vdev->next_seqno;

   handle = vdev->funcs->bo_create(vdev, size, blob_flags, blob_id,
                                   blob_hints, req);

   simple_mtx_unlock(&vdev->eb_lock);

   return handle;
}

#ifdef _WIN32
uint32_t
vdrm_bo_create_shared(struct vdrm_device *vdev, size_t size, uint32_t blob_flags,
                      uint64_t blob_id, uint32_t blob_hints,
                      struct vdrm_ccmd_req *req, void *hRTResource,
                      uint32_t *out_km_resource)
{
   uint32_t handle;

   if (out_km_resource)
      *out_km_resource = 0;

   simple_mtx_lock(&vdev->eb_lock);

   /* Same ordering requirement as vdrm_bo_create: buffered cmds must reach the
    * host before the ones tied to this BO. */
   vdev->funcs->flush_locked(vdev, NULL);

   req->seqno = ++vdev->next_seqno;

   /* Handed to the backend under eb_lock and cleared by it. */
   vdev->pending_shared_rt_resource = hRTResource;
   vdev->pending_shared_km_resource = 0;

   handle = vdev->funcs->bo_create(vdev, size, blob_flags, blob_id,
                                   blob_hints, req);

   if (handle && out_km_resource)
      *out_km_resource = vdev->pending_shared_km_resource;

   /* Clear unconditionally: a backend that bailed before consuming them must
    * not leave a stale resource handle for the next plain vdrm_bo_create. */
   vdev->pending_shared_rt_resource = NULL;
   vdev->pending_shared_km_resource = 0;

   simple_mtx_unlock(&vdev->eb_lock);

   return handle;
}
#endif

void *
vdrm_alloc_rsp(struct vdrm_device *vdev, struct vdrm_ccmd_req *req, uint32_t sz)
{
   unsigned off;

   simple_mtx_lock(&vdev->rsp_lock);

   sz = align(sz, 8);

   if ((vdev->next_rsp_off + sz) >= vdev->rsp_mem_len)
      vdev->next_rsp_off = 0;

   off = vdev->next_rsp_off;
   vdev->next_rsp_off += sz;

   simple_mtx_unlock(&vdev->rsp_lock);

   req->rsp_off = off;

   struct vdrm_ccmd_rsp *rsp = (void *)&vdev->rsp_mem[off];
   rsp->len = sz;

   return rsp;
}

static void vdrm_host_sync_bounded(struct vdrm_device *vdev,
                                   const struct vdrm_ccmd_req *req,
                                   uint32_t timeout_ms);
static void vdrm_host_sync_drain(struct vdrm_device *vdev);

static int
enqueue_req(struct vdrm_device *vdev, struct vdrm_ccmd_req *req)
{
   simple_mtx_assert_locked(&vdev->eb_lock);

   req->seqno = ++vdev->next_seqno;

   if ((vdev->reqbuf_len + req->len) > sizeof(vdev->reqbuf)) {
      int ret = vdev->funcs->flush_locked(vdev, NULL);
      if (ret)
         return ret;
   }

   memcpy(&vdev->reqbuf[vdev->reqbuf_len], req, req->len);
   vdev->reqbuf_len += req->len;
   vdev->reqbuf_cnt++;

   return 0;
}

int
vdrm_execbuf(struct vdrm_device *vdev, struct vdrm_execbuf_params *p)
{
   int ret = 0;

   MESA_TRACE_FUNC();

   /* Never submit new work before the host consumed what we deferred. */
   vdrm_host_sync_drain(vdev);

   simple_mtx_lock(&vdev->eb_lock);

   p->req->seqno = ++vdev->next_seqno;

   ret = vdev->funcs->flush_locked(vdev, NULL);
   if (ret)
      goto out_unlock;

   ret = vdev->funcs->execbuf_locked(vdev, p, p->req, p->req->len);

out_unlock:
   simple_mtx_unlock(&vdev->eb_lock);

   return ret;
}

/**
 * Buffer/send a request cmd to host
 */
int
vdrm_send_req(struct vdrm_device *vdev, struct vdrm_ccmd_req *req, bool sync)
{
   MESA_TRACE_FUNC();

   uintptr_t fence = 0;
   int ret = 0;

   /* Never overwrite the request buffer before the host consumed it. */
   vdrm_host_sync_drain(vdev);

   simple_mtx_lock(&vdev->eb_lock);
   ret = enqueue_req(vdev, req);

   if (ret || !sync)
      goto out_unlock;

   ret = vdev->funcs->flush_locked(vdev, &fence);

out_unlock:
   simple_mtx_unlock(&vdev->eb_lock);

   if (ret)
      return ret;

   if (sync) {
      MESA_TRACE_SCOPE("vdrm_execbuf sync");
      vdev->funcs->wait_fence(vdev, fence);
      vdrm_host_sync(vdev, req);
   }

   return 0;
}

int
vdrm_flush(struct vdrm_device *vdev)
{
   int ret = 0;

   MESA_TRACE_FUNC();

   simple_mtx_lock(&vdev->eb_lock);
   ret = vdev->funcs->flush_locked(vdev, NULL);
   simple_mtx_unlock(&vdev->eb_lock);

   return ret;
}

/**
 * Helper for fence/seqno comparisions which deals properly with rollover.
 * Returns true if fence 'a' is before fence 'b'
 */
static bool
fence_before(uint32_t a, uint32_t b)
{
   return (int32_t)(a - b) < 0;
}

/**
 * Wait until host has processed the specified request.
 */
/* Variant H': the host publishes its progress in shmem->seqno -- shared memory
 * this process only ever reads.  Two defects made an unobserved update fatal:
 * the load was a plain read with no atomicity, and the wait had no bound at
 * all, so if the update was never seen the guest spun forever and wedged the
 * whole VM (observed: 1.2e9 spins, host had already published the seqno).
 * Read atomically and bound the wait; on timeout log and give up, exactly like
 * vsub_sync()/vsub_wait_paging() in vdrm_wddm.c. */
#define VDRM_HOST_SYNC_TIMEOUT_MS 2        /* frame budget: never stall the render thread */
#define VDRM_HOST_SYNC_BLOCKING_TIMEOUT_MS 1000 /* blocking call sites (BO mmap) */
#define VDRM_HOST_SYNC_DRAIN_TIMEOUT_MS 20 /* bound for the deferred drain before the next enqueue */
#define VDRM_HOST_SYNC_STREAK_TIMEOUT_MS 16 /* one frame: re-sync after a consecutive miss */

static void
vdrm_host_sync_bounded(struct vdrm_device *vdev, const struct vdrm_ccmd_req *req,
                     uint32_t timeout_ms)
{
   uint64_t spins = 0;
   /* Sleep between polls: never busy-spin here.  This code runs on a guest
    * vCPU, and the seqno we are waiting for can only advance when the host's
    * single command-dispatch thread gets CPU -- a thrd_yield() spin loop
    * fights that thread for the CPU and makes the wait longer, not shorter
    * (observed: 5-9M spins/s while the host was only 1-8 seqnos behind). */
   uint32_t sleep_us = 20;
   const int64_t deadline =
      os_time_get_nano() + (int64_t)timeout_ms * 1000000ll;

   while (fence_before(p_atomic_read(&vdev->shmem->seqno), req->seqno)) {
      if (unlikely(os_time_get_nano() > deadline)) {
         /* Give-up logging is rate limited: with a ~one-frame bound this can
          * trigger often, and one log line per event would flood stderr. */
         static uint32_t giveups;
         if (giveups++ == 0 || (giveups & 0x3ff) == 0) {
            mesa_loge("vdrm: host not consumed seqno %u within %dms "
                      "(host seqno %u, spins %llu) -- giving up instead of "
                      "sleeping forever",
                      req->seqno, timeout_ms,
                      p_atomic_read(&vdev->shmem->seqno),
                      (unsigned long long)spins);
         }
         return;
      }
      os_time_sleep(sleep_us);
      sleep_us = MIN2(sleep_us + 20, 400);
      spins++;
   }
}

/**
 * Deferred host sync (variant J2).
 *
 * The render thread must never block on the host: on this driver the host's
 * single command-dispatch thread only ever falls a few seqnos behind (measured:
 * host consumed 1-8 seqnos late, ~33us per command), yet a blocking wait turns
 * every one of those microsecond-scale misses into a whole-frame (or worse)
 * stall -- measured by the user as "one freeze per second, a few frames each"
 * with a 1000ms bound.
 *
 * So wait at most a frame budget here.  If the host has still not consumed the
 * request, remember the seqno and return; vdrm_host_sync_drain() re-establishes
 * the ordering guarantee right before the next enqueue (memcpy into reqbuf /
 * execbuf_locked), because nothing after seqno N may be submitted before the
 * host consumed N.  By then the host has almost always caught up, so the drain
 * costs nothing in practice.
 */
void
vdrm_host_sync(struct vdrm_device *vdev, const struct vdrm_ccmd_req *req)
{
   /* Consecutive-deferral guard.  A single unsynced frame is invisible, but a
    * run of them is exactly what showed up as the occasional corrupt frame
    * (the log held two deferrals, matching the two artefacts seen).  Never
    * defer twice in a row: spend one frame budget instead of letting another
    * unsynced frame through. */
   static uint32_t defer_streak;

   if (!fence_before(p_atomic_read(&vdev->shmem->seqno), req->seqno)) {
      defer_streak = 0;               /* already consumed: nothing to do */
      return;
   }

   vdrm_host_sync_bounded(vdev, req, VDRM_HOST_SYNC_TIMEOUT_MS);

   if (!fence_before(p_atomic_read(&vdev->shmem->seqno), req->seqno)) {
      defer_streak = 0;               /* consumed within the frame budget */
      return;
   }

   if (defer_streak++ > 0) {
      /* Second consecutive miss: re-establish the ordering guarantee now,
       * bounded by one frame, instead of deferring again. */
      vdrm_host_sync_bounded(vdev, req, VDRM_HOST_SYNC_STREAK_TIMEOUT_MS);
      defer_streak = 0;

      if (fence_before(p_atomic_read(&vdev->shmem->seqno), req->seqno) &&
          fence_before(vdev->pending_sync_seqno, req->seqno))
         vdev->pending_sync_seqno = req->seqno;

      static uint32_t streaks;
      if (streaks++ == 0 || (streaks & 0x3ff) == 0) {
         mesa_loge("vdrm: consecutive host sync miss, waited %dms for seqno %u "
                   "(host seqno %u)", VDRM_HOST_SYNC_STREAK_TIMEOUT_MS,
                   req->seqno, p_atomic_read(&vdev->shmem->seqno));
      }
      return;
   }

   /* First miss: defer once and keep the highest outstanding watermark. */
   if (fence_before(vdev->pending_sync_seqno, req->seqno))
      vdev->pending_sync_seqno = req->seqno;

   static uint32_t deferrals;
   if (deferrals++ == 0 || (deferrals & 0x3ff) == 0) {
      mesa_loge("vdrm: deferred host sync past seqno %u "
                "(host seqno %u, %dms budget) -- not blocking the frame",
                req->seqno, p_atomic_read(&vdev->shmem->seqno),
                VDRM_HOST_SYNC_TIMEOUT_MS);
   }
}

/**
 * Blocking variant, for call sites that cannot defer: waits (bounded) until the
 * host consumed @req->seqno, so the guest may then safely touch the resource.
 */
void
vdrm_host_sync_blocking(struct vdrm_device *vdev, const struct vdrm_ccmd_req *req)
{
   (void)vdrm_host_sync_bounded(vdev, req, VDRM_HOST_SYNC_BLOCKING_TIMEOUT_MS);
}


static void
vdrm_host_sync_drain(struct vdrm_device *vdev)
{
   uint32_t pending = vdev->pending_sync_seqno;

   if (!pending)
      return;

   if (fence_before(p_atomic_read(&vdev->shmem->seqno), pending)) {
      struct vdrm_ccmd_req req = { .seqno = pending };
      vdrm_host_sync_bounded(vdev, &req, VDRM_HOST_SYNC_DRAIN_TIMEOUT_MS);
   }

   if (!fence_before(p_atomic_read(&vdev->shmem->seqno), vdev->pending_sync_seqno))
      vdev->pending_sync_seqno = 0;
}
