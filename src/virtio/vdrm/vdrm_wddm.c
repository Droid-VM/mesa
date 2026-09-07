/*
 * Copyright © 2026 DroidVM contributors
 * SPDX-License-Identifier: MIT
 *
 * vdrm transport backend for Windows on ARM64（win3d / DroidVM）。
 *
 * Linux 上 vdrm 通过 guest 内核 virtio-gpu DRM 驱动跟 host 通信（vdrm_virtgpu.c）；
 * Windows guest 没有 /dev/dri：这里用 D3DKMT 直连 WDDM 2.0 KMD
 * virtio_wddm_rs.sys（其背后是同一个 host virtio-gpu，capset=DRM(6)）。
 *
 * 协议契约见 win3d/kmd-contract.md（每条都有 KMD 源码行号证据），要点：
 *  - tag = 8 字符 ASCII 小端装入 u64；escape 输入输出共用缓冲、就地回写；
 *  - 适配器识别 = D3DKMTQueryAdapterInfo(UMDRIVERPRIVATE) 的 26B AdapterInfo；
 *  - capset6 = host 的 virgl_renderer_capset_drm（context_type==MSM 才继续）；
 *  - 顺序：CreateDevice → CreateContext → ContextInit escape → 之后才能提交；
 *    CPU map = BlobMap escape（HOST3D|MAPPABLE blob）；
 *  - 单次 escape 恰好一个 CommandHeader；Submit body 实用上限 4064 字节
 *    （KMD 单页拷贝缓冲，不是 uapi 公布的 8192，见 VDRM_WDDM_MAX_SUBMIT_BODY）；
 *  - fence_id 是 EXEC_BUF escape 的 *输出*；KMD 没有"查 escape 提交完成"的
 *    escape（kmd-contract F.2 缺口）—— 这条对 ccmd 无影响：完成信号是
 *    shmem->seqno，见 wddm_wait_fence 的注释。
 *
 * 进度（对应 win3d/vdrm-turnip-windows-移植方案.md 的块）：
 *  块 1：能编（connect 返回 NULL）—— 完成；
 *  块 2：connect 完整实现（枚举→capset→device/context→ContextInit），
 *        vtable 1–5（bo_create/bo_map/bo_close/bo_wait/handle_to_res_id）—— 完成；
 *  块 3：shmem ring（blob_id=0 的 HOST3D|MAPPABLE blob + BlobMap，init_shmem）
 *        —— 完成；
 *  块 4：MAP_BLOB base —— 只在 host 开了 drm2kgsl arena（crosvm 的
 *        --pre-alloc drm-host-mb 注入 CROSVM_DRM2KGSL_ARENA_*）时才要改 KMD：
 *        arena 模式下 ring 从 Drm2KgslPool 切，crosvm 回 map_info|MAP_INFO_POOL，
 *        KMD 认不出直接 IO_DEVICE_ERROR。不开 arena 时走 host 通用 memfd 路径，
 *        KMD 现有 BlobMap 就够（drm2kgsl_renderer.c:2229 的二选一）。
 *  块 5：execbuf/flush + bo_create 的 ccmd 前置提交 —— 本文件当前状态。
 *        已知边界：单条 ccmd 超 4064 字节切不动（MSM_CCMD_GEM_SUBMIT 必然会超，
 *        见 submit_ccmd_stream 的注释），根治要改 KMD 的单页限制。
 *  块 6+：WSI/Present、跨进程共享（dmabuf vtable 仍是 error）。
 *
 * 未确认项（真机验证前别依赖，kmd-contract 未确认 8）：
 *  纯 escape 客户端（从不 render/present）的分配何时被 dxgkrnl open 到设备、
 *  BlobMap 前 host 侧 blob 实体是否已建 —— 代码只证明"attach 时才
 *  RESOURCE_CREATE_BLOB"。
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>

#include "util/os_time.h"
#include "util/u_debug.h"
#include "c11/threads.h"   /* mtx_plain（simple_mtx_init 的 type 参数） */

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
/* d3dkmthk.h 需要 NTSTATUS；这套组合与 win3d UMD / sunflower tu_knl_wddm 一致。 */
#ifndef UMDF_USING_NTSTATUS
#define UMDF_USING_NTSTATUS
#endif
#include <windows.h>
#include <winternl.h>

#include "virtio_wddm_uapi.h"   /* -I$WDDM_INC（build_turnip.sh 会同步此头） */

#include "vdrm.h"

#ifndef NT_SUCCESS
#define NT_SUCCESS(status) (((NTSTATUS)(status)) >= 0)
#endif

/* 句柄表的初始容量与硬上限。上限只为防跑飞，不是设计容量：真实 D3D11 应用的
 * BO 数轻松过千，表按需翻倍。 */
#define VDRM_WDDM_ALLOCS_INITIAL 256
#define VDRM_WDDM_MAX_ALLOCS (1u << 20)
/* ring 大小，与 vdrm_virtgpu.c 的 SHMEM_SZ 一致（KMD 的 shmem offset 分配器按页
 * 记账，0x4000 = 4 页）。 */
#define VDRM_WDDM_SHMEM_SZ 0x4000
/* 与 tu_knl_drm_virtio.cc 里 Windows 的 MAP_FAILED 约定一致（sys/mman.h
 * 不存在于 mingw）。 */
#define VDRM_WDDM_MAP_FAILED ((void *)(uintptr_t)~0ull)
/* 单包 Submit body 的字节上限。KMD 把 `CmdSubmit3d`(32B) + body 拷进一个**单页**
 * 缓冲，超页直接 STATUS_NO_MEMORY（queue.rs:389-414 的 try_from_hdr_with_body，
 * 日志 "cannot handle multi-page boxes yet"）→ 4096 - 32 = 4064。
 * uapi 公布的 MAX_SUBMIT_COMMAND_VIRTUAL_SIZE=8192 是"虚拟上下文私有数据额度"，
 * 不是这条路的真实上限（kmd-contract E.3）。
 * CmdSubmit3d = CtrlHeader(24) + size(4) + padding(4)，见 KMD 的
 * virtio-drivers/src/device/gpu/commands.rs:127-134,391-395。 */
#define VDRM_WDDM_MAX_SUBMIT_BODY 4064

DEBUG_GET_ONCE_BOOL_OPTION(vdrm_wddm_debug, "VDRM_WDDM_DEBUG", false)
#define WDDM_DEBUG(...) do { \
   if (debug_get_option_vdrm_wddm_debug()) \
      fprintf(stderr, "vdrm-wddm: " __VA_ARGS__); \
} while (0)

/* ------------------------------------------------------------------ */
/* wire 布局自检。KMD 对同一批结构有编译期断言（gpu-wddm uapi.rs:1031-1047），
 * 这里逐条对上：aarch64-mingw 的 packed 布局只要错一个字节，每个 escape 都在
 * 静默错位读写，真机上几乎无法定位。期望值来源 win3d/kmd-contract.md 附录 B。 */

_Static_assert(sizeof(VIRTIO_WDDM_AdapterInfo) == 26, "AdapterInfo");
_Static_assert(sizeof(VIRTIO_WDDM_Capset) == 16, "Capset");
_Static_assert(sizeof(VIRTIO_WDDM_ContextInit) == 80, "ContextInit");
_Static_assert(sizeof(VIRTIO_WDDM_ResourceInfo) == 148, "ResourceInfo");
_Static_assert(sizeof(VIRTIO_WDDM_ResourceBusy) == 22, "ResourceBusy");
_Static_assert(sizeof(VIRTIO_WDDM_BlobInfoSet) == 72, "BlobInfoSet");
_Static_assert(sizeof(VIRTIO_WDDM_BlobMap) == 24, "BlobMap");
_Static_assert(sizeof(VIRTIO_WDDM_ExecBuffer) == 16, "ExecBuffer");
_Static_assert(sizeof(VIRTIO_WDDM_Escape) == 148, "Escape union");
_Static_assert(sizeof(VIRTIO_WDDM_Allocate3d) == 56, "Allocate3d");
_Static_assert(sizeof(VIRTIO_WDDM_AllocateBlob) == 32, "AllocateBlob");
_Static_assert(sizeof(VIRTIO_WDDM_CreateAllocation) == 56, "CreateAllocation");
_Static_assert(sizeof(VIRTIO_WDDM_CreateResource) == 8, "CreateResource");
_Static_assert(sizeof(VIRTIO_WDDM_CommandHeader) == 8, "CommandHeader");
_Static_assert(sizeof(VIRTIO_WDDM_CommandTransfer) == 48, "CommandTransfer");
_Static_assert(sizeof(VIRTIO_WDDM_Box3D) == 24, "Box3D");
_Static_assert(sizeof(VIRTIO_WDDM_SubmitCommand) == 8, "SubmitCommand");

/* 关键偏移：KMD 按固定偏移切载荷（EXEC_BUF 的 command_slice 从 16 起，
 * adapter.rs:1770-1773；capset 数据同样紧跟 16 字节头）。 */
_Static_assert(offsetof(VIRTIO_WDDM_ExecBuffer, cmd) == 16, "ExecBuffer.cmd");
_Static_assert(offsetof(VIRTIO_WDDM_Capset, capset) == 16, "Capset.capset");
_Static_assert(offsetof(VIRTIO_WDDM_BlobMap, ptr) == 16, "BlobMap.ptr");
_Static_assert(offsetof(VIRTIO_WDDM_ResourceBusy, is_busy) == 21, "ResourceBusy.is_busy");
_Static_assert(offsetof(VIRTIO_WDDM_CommandHeader, size) == 4, "CommandHeader.size");

/* ring 头（host 与 Linux guest 共用的 wire 结构，drm_hw.h）。 */
_Static_assert(sizeof(struct vdrm_shmem) == 8, "vdrm_shmem");

/* ------------------------------------------------------------------ */

struct vdrm_wddm_dispatch {
   HMODULE gdi32;

   PFND3DKMT_ENUMADAPTERS2 EnumAdapters2;
   PFND3DKMT_OPENADAPTERFROMLUID OpenAdapterFromLuid;
   PFND3DKMT_CLOSEADAPTER CloseAdapter;
   PFND3DKMT_QUERYADAPTERINFO QueryAdapterInfo;
   PFND3DKMT_CREATEDEVICE CreateDevice;
   PFND3DKMT_DESTROYDEVICE DestroyDevice;
   PFND3DKMT_CREATECONTEXT CreateContext;
   PFND3DKMT_DESTROYCONTEXT DestroyContext;
   PFND3DKMT_ESCAPE Escape;
   PFND3DKMT_CREATEALLOCATION2 CreateAllocation;
   PFND3DKMT_DESTROYALLOCATION DestroyAllocation;
};

struct vdrm_wddm_alloc {
   bool used;
   uint32_t next_free;     /* 1-based 空闲链的下一项，0 = 链尾；仅 !used 时有效 */
   D3DKMT_HANDLE kmt;      /* D3DDDI_ALLOCATIONINFO2.hAllocation（KMD 反查句柄） */
   uint32_t res_id;        /* ResourceInfo 返回的 virtio-gpu resource id */
   uint64_t size;
   void *map;              /* BlobMap 返回的 CPU 指针（unmap 时原样带回） */
};

struct vdrm_wddm {
   struct vdrm_device base;

   struct vdrm_wddm_dispatch dispatch;
   LUID luid;
   D3DKMT_HANDLE h_adapter;
   D3DKMT_HANDLE h_device;
   D3DKMT_HANDLE h_context;

   /* BO 句柄表。Linux 后端把句柄交给 drm/virtio 内核管，Windows 这边得自己管。
    * 必须自带锁：turnip 会从多个应用线程 create/map/close BO，而 vdrm.c 只在
    * bo_create 期间持 eb_lock（vdrm.c:57-70），bo_map/bo_close/handle_to_res_id
    * 都是裸调用。句柄 = 表下标 + 1（0 保留作无效）；表 realloc 增长不影响已发出
    * 的句柄，因为句柄是下标而不是指针。 */
   simple_mtx_t table_lock;
   struct vdrm_wddm_alloc *allocs;
   uint32_t allocs_capacity;
   uint32_t free_head;        /* 1-based，0 = 无空闲槽 */

   uint32_t shmem_handle;     /* ring 的表句柄（回收由 wddm_close 的通用路径做） */
};

/* ------------------------------------------------------------------ */
/* 小工具                                                                */

static NTSTATUS
wddm_escape(struct vdrm_wddm *w, D3DKMT_HANDLE h_device, D3DKMT_HANDLE h_context,
            void *buf, uint32_t buf_size)
{
   D3DKMT_ESCAPE escape = {0};

   escape.hAdapter = w->h_adapter;
   escape.hDevice = h_device;
   escape.hContext = h_context;
   escape.Type = D3DKMT_ESCAPE_DRIVERPRIVATE;
   escape.pPrivateDriverData = buf;
   escape.PrivateDriverDataSize = buf_size;

   NTSTATUS status = w->dispatch.Escape(&escape);
   WDDM_DEBUG("Escape %.8s: status=0x%08lx\n", (const char *)buf, (ULONG)status);
   return status;
}

static void
wddm_destroy_alloc(struct vdrm_wddm *w, D3DKMT_HANDLE kmt)
{
   D3DKMT_DESTROYALLOCATION destroy = {0};

   destroy.hDevice = w->h_device;
   destroy.phAllocationList = &kmt;
   destroy.AllocationCount = 1;
   w->dispatch.DestroyAllocation(&destroy);
}

/* ------------------------------------------------------------------ */
/* 句柄表（全部在 table_lock 下；不把表项指针交出去，realloc 之后会失效）      */

/* 占一个槽并填入。返回 1-based 句柄，0 = 失败。 */
static uint32_t
table_insert(struct vdrm_wddm *w, D3DKMT_HANDLE kmt, uint32_t res_id, uint64_t size)
{
   uint32_t handle = 0;

   simple_mtx_lock(&w->table_lock);

   if (!w->free_head) {
      uint32_t old_cap = w->allocs_capacity;
      uint32_t new_cap = old_cap ? old_cap * 2 : VDRM_WDDM_ALLOCS_INITIAL;
      struct vdrm_wddm_alloc *grown;
      uint32_t i;

      if (new_cap > VDRM_WDDM_MAX_ALLOCS)
         goto out;

      grown = realloc(w->allocs, new_cap * sizeof(*grown));
      if (!grown)
         goto out;

      memset(&grown[old_cap], 0, (new_cap - old_cap) * sizeof(*grown));
      w->allocs = grown;
      w->allocs_capacity = new_cap;

      /* 新槽穿成空闲链，低下标在链头（句柄分配尽量紧凑）。 */
      for (i = new_cap; i > old_cap; i--) {
         grown[i - 1].next_free = w->free_head;
         w->free_head = i;
      }
   }

   handle = w->free_head;
   w->free_head = w->allocs[handle - 1].next_free;

   w->allocs[handle - 1].used = true;
   w->allocs[handle - 1].next_free = 0;
   w->allocs[handle - 1].kmt = kmt;
   w->allocs[handle - 1].res_id = res_id;
   w->allocs[handle - 1].size = size;
   w->allocs[handle - 1].map = NULL;

out:
   simple_mtx_unlock(&w->table_lock);
   return handle;
}

/* 读出一份拷贝。 */
static bool
table_get(struct vdrm_wddm *w, uint32_t handle, struct vdrm_wddm_alloc *out)
{
   bool ok = false;

   simple_mtx_lock(&w->table_lock);
   if (handle && handle <= w->allocs_capacity && w->allocs[handle - 1].used) {
      *out = w->allocs[handle - 1];
      ok = true;
   }
   simple_mtx_unlock(&w->table_lock);

   return ok;
}

/* 摘掉槽并把内容交出来（归还空闲链）。摘掉之后别的线程再也查不到这个句柄，
 * 所以调用者可以在锁外做 unmap/DestroyAllocation。 */
static bool
table_remove(struct vdrm_wddm *w, uint32_t handle, struct vdrm_wddm_alloc *out)
{
   bool ok = false;

   simple_mtx_lock(&w->table_lock);
   if (handle && handle <= w->allocs_capacity && w->allocs[handle - 1].used) {
      *out = w->allocs[handle - 1];
      memset(&w->allocs[handle - 1], 0, sizeof(w->allocs[handle - 1]));
      w->allocs[handle - 1].next_free = w->free_head;
      w->free_head = handle;
      ok = true;
   }
   simple_mtx_unlock(&w->table_lock);

   return ok;
}

/* ------------------------------------------------------------------ */
/* 初始化序列（契约附录 D）：枚举→capset→device→context→ContextInit          */

static bool
dispatch_init(struct vdrm_wddm_dispatch *d)
{
   const struct {
      const char *name;
      void **slot;
   } thunks[] = {
      { "D3DKMTEnumAdapters2",       (void **)&d->EnumAdapters2 },
      { "D3DKMTOpenAdapterFromLuid", (void **)&d->OpenAdapterFromLuid },
      { "D3DKMTCloseAdapter",        (void **)&d->CloseAdapter },
      { "D3DKMTQueryAdapterInfo",    (void **)&d->QueryAdapterInfo },
      { "D3DKMTCreateDevice",        (void **)&d->CreateDevice },
      { "D3DKMTDestroyDevice",       (void **)&d->DestroyDevice },
      { "D3DKMTCreateContext",       (void **)&d->CreateContext },
      { "D3DKMTDestroyContext",      (void **)&d->DestroyContext },
      { "D3DKMTEscape",              (void **)&d->Escape },
      /* The *2 thunk consumes D3DDDI_ALLOCATIONINFO2 below. */
      { "D3DKMTCreateAllocation2",   (void **)&d->CreateAllocation },
      { "D3DKMTDestroyAllocation",   (void **)&d->DestroyAllocation },
   };
   size_t i;

   memset(d, 0, sizeof(*d));
   d->gdi32 = LoadLibraryA("gdi32.dll");
   if (!d->gdi32)
      return false;

   for (i = 0; i < sizeof(thunks) / sizeof(thunks[0]); i++) {
      *thunks[i].slot = (void *)GetProcAddress(d->gdi32, thunks[i].name);
      if (!*thunks[i].slot) {
         WDDM_DEBUG("missing %s (error=%lu)\n", thunks[i].name, GetLastError());
         return false;
      }
   }
   return true;
}

static void
dispatch_fini(struct vdrm_wddm_dispatch *d)
{
   if (d->gdi32)
      FreeLibrary(d->gdi32);
   memset(d, 0, sizeof(*d));
}

/* 适配器私有自检：tag/supports_3d/has_shmem，要求 DRM capset（对照 win3d
 * UMD adapter.c:130-152 的校验）。 */
static bool
query_adapter_info(struct vdrm_wddm_dispatch *d, D3DKMT_HANDLE h_adapter,
                   VIRTIO_WDDM_AdapterInfo *out)
{
   D3DKMT_QUERYADAPTERINFO query = {0};
   NTSTATUS status;

   memset(out, 0, sizeof(*out));
   query.hAdapter = h_adapter;
   query.Type = KMTQAITYPE_UMDRIVERPRIVATE;
   query.pPrivateDriverData = out;
   query.PrivateDriverDataSize = sizeof(*out);

   status = d->QueryAdapterInfo(&query);
   if (!NT_SUCCESS(status))
      return false;

   return out->tag == VIRTIO_WDDM_ADAPTER_INFO_TAG &&
          out->supports_3d && out->has_shmem &&
          (out->capset_mask & VIRTIO_WDDM_CAPSET_MASK_DRM);
}

/* 读 capset：输出就地写在 Capset 头(16B)之后；version 被 KMD 忽略；字节数 =
 * min(缓冲剩余, host 长度)，静默截断。 */
static bool
query_capset(struct vdrm_wddm *w, D3DKMT_HANDLE h_device, VIRTIO_WDDM_CapsetId id,
             void *data_out, uint32_t data_out_size)
{
   struct {
      VIRTIO_WDDM_Capset hdr;
      uint8_t data[512];
   } buf;
   NTSTATUS status;
   uint32_t got;

   memset(&buf, 0, sizeof(buf));
   buf.hdr.tag = VIRTIO_WDDM_ESCAPE_CAPSET_TAG;
   buf.hdr.capset_id = id;
   buf.hdr.version = 0;

   status = wddm_escape(w, h_device, 0, &buf, sizeof(buf));
   if (!NT_SUCCESS(status))
      return false;

   got = (uint32_t)sizeof(buf) - (uint32_t)sizeof(VIRTIO_WDDM_Capset);
   if (got > data_out_size)
      got = data_out_size;
   memcpy(data_out, buf.data, got);
   return true;
}

static bool
context_init(struct vdrm_wddm *w)
{
   VIRTIO_WDDM_ContextInit ctx = {0};
   NTSTATUS status;

   ctx.tag = VIRTIO_WDDM_ESCAPE_CONTEXT_INIT_TAG;
   ctx.capset_id = VIRTIO_WDDM_CAPSET_ID_DRM;
   ctx.num_rings = 1;   /* KMD 不使用 num_rings；host 多 ring 语义未确认 */
   strncpy((char *)ctx.debug_name, "vdrm-turnip-win32", sizeof(ctx.debug_name) - 1);

   status = wddm_escape(w, w->h_device, 0, &ctx, sizeof(ctx));
   return NT_SUCCESS(status);
}

static void
wddm_close_handles(struct vdrm_wddm *w)
{
   if (w->h_context) {
      D3DKMT_DESTROYCONTEXT destroy = {0};
      destroy.hContext = w->h_context;
      if (NT_SUCCESS(w->dispatch.DestroyContext(&destroy)))
         w->h_context = 0;
   }
   if (w->h_device) {
      D3DKMT_DESTROYDEVICE destroy = {0};
      destroy.hDevice = w->h_device;
      if (NT_SUCCESS(w->dispatch.DestroyDevice(&destroy)))
         w->h_device = 0;
   }
   if (w->h_adapter) {
      D3DKMT_CLOSEADAPTER close = {0};
      close.hAdapter = w->h_adapter;
      if (NT_SUCCESS(w->dispatch.CloseAdapter(&close)))
         w->h_adapter = 0;
   }
}

/* ------------------------------------------------------------------ */
/* 提交（块 5）：ccmd 流 → EXEC_BUF escape                                  */

/* 一包 = ExecBuffer 头(16) + 恰好一个 CommandHeader(8) + body。
 * KMD 校验 `8 + header.size == cmd.len()`，多一条命令就 INVALID_PARAMETER
 * （adapter.rs:1775-1801），所以每次 escape 只发一个 CommandHeader。
 * body 对 KMD 不透明，msm ccmd 流原样塞进去即可（kmd-contract G 第 5 条）。
 * fence_id 是**输出**（adapter.rs:1822），入参填 0。
 * hContext 必填且其 Device 必须已 ContextInit（adapter.rs:1808-1809）。 */
static int
submit_one(struct vdrm_wddm *w, const void *body, uint32_t body_len, uint8_t ring)
{
   const uint32_t body_off = offsetof(VIRTIO_WDDM_ExecBuffer, cmd) +
                             sizeof(VIRTIO_WDDM_CommandHeader);
   uint8_t buf[offsetof(VIRTIO_WDDM_ExecBuffer, cmd) +
               sizeof(VIRTIO_WDDM_CommandHeader) + VDRM_WDDM_MAX_SUBMIT_BODY];
   VIRTIO_WDDM_CommandHeader hdr = {0};
   VIRTIO_WDDM_ExecBuffer eb = {0};
   NTSTATUS status;

   if (body_len == 0 || body_len > VDRM_WDDM_MAX_SUBMIT_BODY)
      return -EINVAL;

   eb.tag = VIRTIO_WDDM_ESCAPE_EXEC_BUF_TAG;
   eb.fence_id = 0;

   hdr.id = VIRTIO_WDDM_COMMAND_ID_SUBMIT;
   /* 与 Linux 后端一致：那边恒置 VIRTGPU_EXECBUF_RING_IDX（vdrm_virtgpu.c 的
    * execbuf），flush 走 ring 0、submit 走 queue->priority+1。KMD 不校验 ring
    * 值，只透传进 virtio CtrlHeader；crosvm 侧 ring 是按需建的 fence ring key
    * （gpu/mod.rs:1062），不需要预先声明，所以 ContextInit 的 num_rings 无关。 */
   hdr.flags = VIRTIO_WDDM_COMMAND_FLAG_RING_IDX;
   hdr.ring = ring;
   hdr.size = body_len;

   /* escape 缓冲就地回写，逐段 memcpy 组包（wire 结构 packed，不做对齐假设，
    * kmd-contract I.3）。 */
   memcpy(buf, &eb, offsetof(VIRTIO_WDDM_ExecBuffer, cmd));
   memcpy(buf + offsetof(VIRTIO_WDDM_ExecBuffer, cmd), &hdr, sizeof(hdr));
   memcpy(buf + body_off, body, body_len);

   status = wddm_escape(w, w->h_device, w->h_context, buf, body_off + body_len);
   if (!NT_SUCCESS(status))
      return -EIO;

   return 0;
}

/* 把一段 ccmd 流发出去，必要时分包。
 *
 * 分包只能按 **ccmd 边界** 切：流里是一串 vdrm_ccmd_req，每条自带 len，host 按条
 * 解析（drm_hw.h:89-140）。按字节切会让 host 从一条命令中间开始解析。
 *
 * 单条 ccmd 自己就超上限时切不动，只能 -E2BIG。这不是边缘情况：
 * MSM_CCMD_GEM_SUBMIT 的 nr_bos = 设备当前**全部**活着的 BO
 * （tu_knl_drm_virtio.cc:1659），每个 drm_msm_gem_submit_bo 16 字节 → 约 250 个 BO
 * 就撞 4064。真实渲染必然超。根治要改 KMD 的单页限制（queue.rs:389-414），属于
 * KMD ARM64 移植那一摊；在那之前 vulkaninfo 一类不提交渲染的路径不受影响。 */
static int
submit_ccmd_stream(struct vdrm_wddm *w, const void *data, uint32_t len, uint8_t ring)
{
   const uint8_t *bytes = (const uint8_t *)data;
   uint32_t off = 0;

   while (off < len) {
      uint32_t chunk = 0;
      int ret;

      /* 攒到再加一条就超上限为止。 */
      while (off + chunk < len) {
         uint32_t req_len;

         if (len - (off + chunk) < sizeof(struct vdrm_ccmd_req))
            return -EINVAL;
         memcpy(&req_len, bytes + off + chunk + offsetof(struct vdrm_ccmd_req, len),
                sizeof(req_len));
         if (req_len < sizeof(struct vdrm_ccmd_req) || req_len > len - (off + chunk))
            return -EINVAL;
         if (chunk && chunk + req_len > VDRM_WDDM_MAX_SUBMIT_BODY)
            break;
         chunk += req_len;
      }

      if (chunk > VDRM_WDDM_MAX_SUBMIT_BODY)
         return -E2BIG;   /* 单条超上限，host 按整条解析，切不动 */

      ret = submit_one(w, bytes + off, chunk, ring);
      if (ret)
         return ret;

      off += chunk;
   }

   return 0;
}

/* ------------------------------------------------------------------ */
/* vtable（vdrm.h 的 vdrm_device_funcs）                                  */

static int
wddm_execbuf_locked(struct vdrm_device *vdev, struct vdrm_execbuf_params *p,
                    void *command, unsigned size)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;

   simple_mtx_assert_locked(&vdev->eb_lock);

   /* Windows 上没有 drm syncobj、没有 sync_file fd。turnip 必须跑
    * userspace-fence（poll）模式；真走到这里说明上层还在用 syncobj/fence fd
    * 做同步，宁可失败也不能静默丢掉同步语义。 */
   if (p->num_in_syncobjs || p->num_out_syncobjs)
      return -ENOSYS;
   if (p->has_in_fence_fd || p->needs_out_fence_fd)
      return -ENOSYS;

   if (p->ring_idx < 0 || p->ring_idx > 255)
      return -EINVAL;

   /* p->handles / num_handles 故意忽略：Linux 上那是给 guest 内核做隐式同步和
    * BO 存活保证用的；escape 型提交不关联任何 allocation（kmd-contract F.2），
    * host 侧 BO 生命周期由 ccmd 流自己描述。 */

   return submit_ccmd_stream(w, command, size, (uint8_t)p->ring_idx);
}

static int
wddm_flush_locked(struct vdrm_device *vdev, uintptr_t *fencep)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   int ret;

   simple_mtx_assert_locked(&vdev->eb_lock);

   /* Linux 这里会要一个 out fence fd 交给 wait_fence。Windows 没有等价物，也不
    * 需要：ccmd 的完成信号是 shmem->seqno，vdrm_send_req(sync=true) 在
    * wait_fence 之后还会调 vdrm_host_sync 轮询 seqno（vdrm.c:167-171），
    * 所以 fence 恒 0 + wait_fence 空实现是正确的组合。 */
   if (fencep)
      *fencep = 0;

   if (!vdev->reqbuf_len)
      return 0;

   /* 与 Linux 的 flush 一致走 ring 0（那边 p.ring_idx 默认 0）。 */
   ret = submit_ccmd_stream(w, vdev->reqbuf, vdev->reqbuf_len, 0);
   if (ret)
      return ret;

   vdev->reqbuf_len = 0;
   vdev->reqbuf_cnt = 0;

   return 0;
}

/* 空实现是正确的，不是占位：flush_locked 恒回 fence 0（Windows 没有 out fence
 * fd），而 ccmd 的真实完成信号是 shmem->seqno —— vdrm_send_req(sync=true) 在
 * wait_fence 之后还会调 vdrm_host_sync 轮询它（vdrm.c:167-171）。KMD 也确实没有
 * "查询/等待某 virtio fence" 的 escape（kmd-contract F.2）。
 * GPU 执行完成（VkFence/VkSemaphore）走 turnip 的 userspace fence，不经这里。 */
static void
wddm_wait_fence(struct vdrm_device *vdev, uintptr_t fence)
{
   (void)vdev; (void)fence;
}

static uint32_t
wddm_dmabuf_to_handle(struct vdrm_device *vdev, int fd)
{
   /* Windows 没有 PRIME/dmabuf（方案缺口 1），块 1–6 直接失败。 */
   (void)vdev; (void)fd;
   return 0;
}

static uint32_t
wddm_handle_to_res_id(struct vdrm_device *vdev, uint32_t handle)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   struct vdrm_wddm_alloc alloc;

   if (!table_get(w, handle, &alloc))
      return 0;
   return alloc.res_id;
}

/* D3DKMTAllocate：资源级私有数据 tag VALLRESR（cmd 空）+ 每分配级
 * blob(HOST3D)。blob_flags（VIRTGPU_BLOB_FLAG_USE_*）与
 * VIRTIO_WDDM_BlobFlag 同值（都源于 virtio-gpu 规范），直接透传。
 * host 侧 blob 实体在 attach 时创建（kmd-contract C.3 / 未确认 8）。
 * 返回 vdrm 句柄（=表下标+1，0=失败）。
 *
 * `req`（BO 描述 ccmd，通常是 MSM_CCMD_GEM_NEW）在 Linux 是随 ioctl **原子**
 * 下发的（vdrm_virtgpu.c:196-197 的 .cmd/.cmd_size）；D3DKMT 没有"分配捎带命令"
 * 的等价物，所以这里先单独 EXEC_BUF 发出去，再 D3DKMTAllocate。
 * 顺序有保证（不需要真机验证）：KMD 的 submit_command_buffer 走
 * self.control.request_async_buf（queue.rs:2144），resource_create_blob 走
 * self.control.request_blocking（queue.rs:1932），而 request_blocking_buf 内部就是
 * request_async_buf + 等 KeEvent（queue.rs:1147-1171）—— 同一个 control channel
 * 的同一条 FIFO；crosvm 侧 SUBMIT_3D 与 RESOURCE_CREATE_BLOB 同在控制队列按序
 * 同步处理，drm renderer 的 ccmd 是 inline 执行。
 * host 侧必须先看到这条 ccmd：drm2kgsl 的 get_blob 对 blob_id!=0 要查到 GEM_NEW
 * 注册的对象，查不到就 -ENOENT（drm2kgsl_renderer.c:1114-1119），还核对 blob_size。 */
static uint32_t
wddm_bo_create(struct vdrm_device *vdev, size_t size, uint32_t blob_flags,
               uint64_t blob_id, uint32_t blob_hints,
               struct vdrm_ccmd_req *req)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   VIRTIO_WDDM_CreateResource res_priv = {0};
   VIRTIO_WDDM_CreateAllocation alloc_priv = {0};
   D3DDDI_ALLOCATIONINFO2 alloc_info = {0};
   D3DKMT_CREATEALLOCATION allocate = {0};
   VIRTIO_WDDM_ResourceInfo res_info = {0};
   NTSTATUS status;
   uint32_t handle;

   (void)blob_hints;

   /* vdrm_bo_create 已经 flush 过 reqbuf 并给 req 分配了 seqno（vdrm.c:57-70），
    * 所以这条一定不在 reqbuf 里，单独发。ring 0，与 flush 同路。 */
   if (req && submit_ccmd_stream(w, req, req->len, 0))
      return 0;

   res_priv.tag = VIRTIO_WDDM_CREATE_RESOURCE_TAG;

   alloc_priv.blob.tag = VIRTIO_WDDM_ALLOCATE_BLOB_TAG;
   alloc_priv.blob.id = blob_id;
   const bool guest_alloc = vdev->supports_guest_alloc && blob_id != 0;
   alloc_priv.blob.mem = guest_alloc ? VIRTIO_WDDM_BLOB_MEM_HOST3D_GUEST : VIRTIO_WDDM_BLOB_MEM_HOST3D;
   if (guest_alloc) blob_flags |= VIRTIO_WDDM_BLOB_FLAG_CREATE_GUEST_HANDLE;
   alloc_priv.blob.flags = (VIRTIO_WDDM_BlobFlag)blob_flags;
   alloc_priv.blob.size = size;

   alloc_info.pPrivateDriverData = &alloc_priv;
   alloc_info.PrivateDriverDataSize = sizeof(alloc_priv);

   allocate.hDevice = w->h_device;
   allocate.pPrivateDriverData = &res_priv;
   allocate.PrivateDriverDataSize = sizeof(res_priv);
   allocate.NumAllocations = 1;
   allocate.pAllocationInfo2 = &alloc_info;
   allocate.Flags.CreateResource = 1;

   status = w->dispatch.CreateAllocation(&allocate);
   WDDM_DEBUG("CreateAllocation blob=%llu size=%zu: status=0x%08lx handle=0x%x\n",
              (unsigned long long)blob_id, size, (ULONG)status, alloc_info.hAllocation);
   if (!NT_SUCCESS(status) || alloc_info.hAllocation == 0)
      return 0;

   /* ResourceInfo：拿 virtio resource id（后续 ccmd 引用资源的句柄）。 */
   res_info.tag = VIRTIO_WDDM_ESCAPE_RESOURCE_INFO_TAG;
   res_info.handle = alloc_info.hAllocation;
   status = wddm_escape(w, w->h_device, 0, &res_info, sizeof(res_info));
   if (!NT_SUCCESS(status) || res_info.id == 0) {
      wddm_destroy_alloc(w, alloc_info.hAllocation);
      return 0;
   }

   /* 建表项。KMD 调用都在锁外做完，这里只占槽。 */
   handle = table_insert(w, alloc_info.hAllocation, res_info.id, size);
   if (!handle) {
      wddm_destroy_alloc(w, alloc_info.hAllocation);
      return 0;
   }

   return handle;
}

static int
wddm_bo_wait(struct vdrm_device *vdev, uint32_t handle)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   struct vdrm_wddm_alloc alloc;
   int64_t deadline = 0;

   if (!table_get(w, handle, &alloc))
      return -EINVAL;

   /* ResourceBusy 衡量的是 render/present DMA 占用；escape 提交路径（块 5 前
    * 不存在）不参与 mark_busy，所以这里几乎总是立即返回。轮询兜底 30s。
    * （kmd-contract F.3：busy 计数只在 render/present DMA 路径变化）
    * 不持 table_lock：这里可能阻塞很久。 */
   for (;;) {
      VIRTIO_WDDM_ResourceBusy busy = {0};
      NTSTATUS status;

      busy.tag = VIRTIO_WDDM_ESCAPE_RESOURCE_BUSY_TAG;
      busy.handle = alloc.kmt;
      busy.wait = false;

      status = wddm_escape(w, w->h_device, 0, &busy, sizeof(busy));
      if (!NT_SUCCESS(status))
         return -EIO;
      if (!busy.is_busy)
         return 0;

      if (!deadline)
         deadline = os_time_get_nano() + (int64_t)30e9;
      if (os_time_get_nano() >= deadline)
         return -ETIMEDOUT;
      Sleep(1);
   }
}

static void *
wddm_bo_map(struct vdrm_device *vdev, uint32_t handle, size_t size, void *placed_addr)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   VIRTIO_WDDM_BlobMap map = {0};
   void *ret = VDRM_WDDM_MAP_FAILED;
   NTSTATUS status;

   (void)size;   /* BlobMap 没有 size 字段，KMD 从 allocation 自己取 */

   if (placed_addr != NULL)
      return VDRM_WDDM_MAP_FAILED;   /* 没有固定地址语义 */

   /* 整段持锁（escape 也在锁内）：KMD 侧一个 blob 同时只能有一个映射，重复 map
    * 回 STATUS_ALREADY_COMMITTED（kmd-contract D.2），所以"查未映射 → 映射 →
    * 记指针"必须原子，否则两个线程 map 同一个 BO 会有一方莫名失败。 */
   simple_mtx_lock(&w->table_lock);

   if (!handle || handle > w->allocs_capacity || !w->allocs[handle - 1].used)
      goto out;
   if (w->allocs[handle - 1].map)
      goto out;   /* 先 unmap 再 map */

   map.tag = VIRTIO_WDDM_ESCAPE_BLOB_MAP_TAG;
   map.handle = w->allocs[handle - 1].kmt;
   map.flags = 0;
   map.ptr.ptr = NULL;

   status = wddm_escape(w, w->h_device, 0, &map, sizeof(map));
   if (!NT_SUCCESS(status) || map.ptr.ptr == NULL)
      goto out;

   w->allocs[handle - 1].map = map.ptr.ptr;
   ret = map.ptr.ptr;

out:
   simple_mtx_unlock(&w->table_lock);
   return ret;
}

static int
wddm_bo_export_dmabuf(struct vdrm_device *vdev, uint32_t handle)
{
   (void)vdev; (void)handle;
   return -1;
}

static void
wddm_bo_close(struct vdrm_device *vdev, uint32_t handle)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   struct vdrm_wddm_alloc alloc;

   /* 先把槽摘掉，escape 在锁外做：摘掉之后别的线程查不到这个句柄了。 */
   if (!table_remove(w, handle, &alloc))
      return;

   if (alloc.map) {
      VIRTIO_WDDM_BlobMap unmap = {0};

      /* UNMAP 必须把当初拿到的 ptr 原样带回（KMD 靠它撤销用户态映射，
       * kmd-contract D.2）。 */
      unmap.tag = VIRTIO_WDDM_ESCAPE_BLOB_MAP_TAG;
      unmap.handle = alloc.kmt;
      unmap.flags = VIRTIO_WDDM_BLOB_MAP_FLAGS_UNMAP;
      unmap.ptr.ptr = alloc.map;
      wddm_escape(w, w->h_device, 0, &unmap, sizeof(unmap));
   }

   wddm_destroy_alloc(w, alloc.kmt);
}

static void
wddm_close(struct vdrm_device *vdev)
{
   struct vdrm_wddm *w = (struct vdrm_wddm *)vdev;
   uint32_t i;

   /* 按依赖顺序：先资源（ring 也在表里，一起回收），再 context/device/adapter。 */
   for (i = 1; i <= w->allocs_capacity; i++)
      wddm_bo_close(vdev, i);

   w->base.shmem = NULL;
   w->base.rsp_mem = NULL;
   w->base.rsp_mem_len = 0;
   w->shmem_handle = 0;

   free(w->allocs);
   w->allocs = NULL;
   w->allocs_capacity = 0;
   w->free_head = 0;
   simple_mtx_destroy(&w->table_lock);

   wddm_close_handles(w);
   dispatch_fini(&w->dispatch);
}

static const struct vdrm_device_funcs vdrm_wddm_funcs = {
   .execbuf_locked = wddm_execbuf_locked,
   .flush_locked = wddm_flush_locked,
   .wait_fence = wddm_wait_fence,
   .dmabuf_to_handle = wddm_dmabuf_to_handle,
   .handle_to_res_id = wddm_handle_to_res_id,
   .bo_create = wddm_bo_create,
   .bo_wait = wddm_bo_wait,
   .bo_map = wddm_bo_map,
   .bo_export_dmabuf = wddm_bo_export_dmabuf,
   .bo_close = wddm_bo_close,
   .close = wddm_close,
};

/* ------------------------------------------------------------------ */
/* 块 3：shmem ring                                                       */

/* ring = vdrm_shmem 头 + 响应区。上游 Linux 是 vdrm_virtgpu.c:353 的 init_shmem：
 * blob_id=0 的 HOST3D|MAPPABLE blob。host drm2kgsl 对 blob_id==0 在对象查找之前
 * 就特判掉了（drm2kgsl_renderer.c:2228-2233），所以这个 blob 不需要先发 ccmd，
 * 正好能用现在还丢 req 的 bo_create 建出来。
 * rsp_mem_offset 由 host 建 blob 时写入（= sizeof(struct msm_shmem)），guest 只读；
 * 响应区的分配由 guest 自己管（vdrm.c 的 vdrm_alloc_rsp）。 */
static bool
init_shmem(struct vdrm_wddm *w)
{
   struct vdrm_device *vdev = &w->base;
   uint32_t handle;
   uint32_t offset;
   void *ptr;

   handle = wddm_bo_create(vdev, VDRM_WDDM_SHMEM_SZ,
                           VIRTIO_WDDM_BLOB_FLAG_MAPPABLE,
                           0 /* blob_id：ring */, 0 /* blob_hints */,
                           NULL /* 没有随行 ccmd */);
   if (!handle)
      return false;

   ptr = wddm_bo_map(vdev, handle, VDRM_WDDM_SHMEM_SZ, NULL);
   if (ptr == VDRM_WDDM_MAP_FAILED) {
      wddm_bo_close(vdev, handle);
      return false;
   }

   offset = ((struct vdrm_shmem *)ptr)->rsp_mem_offset;
   WDDM_DEBUG("ring rsp_mem_offset=%u\n", offset);

   /* Opt-in bringup check: keep a failed ring alive long enough to compare its
    * last word with the host's arena mapping. Restore it before releasing the
    * allocation; this never runs for a valid ring or during normal startup. */
   if (!offset && debug_get_bool_option("VDRM_WDDM_MAP_PROBE", false)) {
      volatile uint32_t *words = ptr;
      volatile uint32_t *tail = words + VDRM_WDDM_SHMEM_SZ / 4 - 1;
      uint32_t saved = *tail;
      WDDM_DEBUG("map probe ptr=%p header=%08x,%08x,%08x,%08x tail=%08x\n",
                 ptr, words[0], words[1], words[2], words[3], saved);
      *tail = 0x44564d31;
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
      WDDM_DEBUG("map probe immediate readback=%08x\n", *tail);
      Sleep(10000);
      WDDM_DEBUG("map probe after wait header=%08x,%08x tail=%08x\n",
                 words[0], words[1], *tail);
      *tail = saved;
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
   }

   /* host 没写（或写歪）时就地失败。offset=0 会让 vdrm_alloc_rsp 把响应头写到
    * seqno 上，症状要拖到很后面才显形。kmd-contract「未确认 8」（纯 escape
    * 客户端的 blob 何时在 host 落地）真出问题，就是这里报出来。 */
   if (offset < sizeof(struct vdrm_shmem) || offset >= VDRM_WDDM_SHMEM_SZ) {
      wddm_bo_close(vdev, handle);
      return false;
   }

   w->shmem_handle = handle;
   vdev->shmem = (struct vdrm_shmem *)ptr;
   vdev->rsp_mem = (uint8_t *)ptr + offset;
   vdev->rsp_mem_len = VDRM_WDDM_SHMEM_SZ - offset;

   return true;
}

/* ------------------------------------------------------------------ */

/* vdrm.c 在 _WIN32 下 forward-declare 本函数；这里自声明满足
 * -Wmissing-prototypes，同时保证 vdrm.h 不泄漏 Windows 专用符号。 */
struct vdrm_device * vdrm_wddm_connect(int fd, uint32_t context_type);

struct vdrm_device *
vdrm_wddm_connect(int fd, uint32_t context_type)
{
   struct vdrm_wddm *w = NULL;
   VIRTIO_WDDM_AdapterInfo adapter_info;
   NTSTATUS status;
   bool ok = false;
   const char *stage = "dispatch";
   ULONG i;

   (void)fd;   /* Windows 没有 fd；保留参数仅为接口一致 */

   w = (struct vdrm_wddm *)calloc(1, sizeof(*w));
   if (!w)
      return NULL;
   w->base.funcs = &vdrm_wddm_funcs;
   /* 在任何 goto out 之前：失败路径统一走 wddm_close()，它会 destroy 这把锁。 */
   simple_mtx_init(&w->table_lock, mtx_plain);

   if (!dispatch_init(&w->dispatch))
      goto out;
   stage = "adapter enumeration";

   /* 1. 枚举适配器找 virtio。EnumAdapters2 返回的句柄用完即关，命中后按
    *    LUID 重开（与 sunflower tu_knl_wddm 同模式）。 */
   {
      D3DKMT_ENUMADAPTERS2 enum2 = {0};
      NTSTATUS s = w->dispatch.EnumAdapters2(&enum2);

      if (NT_SUCCESS(s) && enum2.NumAdapters > 0) {
         D3DKMT_ADAPTERINFO *entries;
         ULONG n = enum2.NumAdapters;

         entries = (D3DKMT_ADAPTERINFO *)calloc(n, sizeof(*entries));
         if (!entries)
            goto out;
         enum2.pAdapters = entries;
         s = w->dispatch.EnumAdapters2(&enum2);
         if (!NT_SUCCESS(s) || enum2.NumAdapters > n) {
            free(entries);
            goto out;
         }

         for (i = 0; i < enum2.NumAdapters; i++) {
            if (!w->luid.HighPart && !w->luid.LowPart && entries[i].hAdapter &&
                query_adapter_info(&w->dispatch, entries[i].hAdapter,
                                   &adapter_info)) {
               w->luid = entries[i].AdapterLuid;
            }
            if (entries[i].hAdapter) {
               D3DKMT_CLOSEADAPTER close = {0};
               close.hAdapter = entries[i].hAdapter;
               w->dispatch.CloseAdapter(&close);
            }
            /* EnumAdapters2 opens every returned handle. Close the remaining
             * entries even after selecting the first matching adapter. */
         }
         free(entries);
      }
   }
   if (!w->luid.HighPart && !w->luid.LowPart)
      goto out;   /* 没找到 virtio 适配器 */

   /* 2. 按 LUID 打开并复验。 */
   stage = "open adapter";
   {
      D3DKMT_OPENADAPTERFROMLUID open = {0};
      open.AdapterLuid = w->luid;
      status = w->dispatch.OpenAdapterFromLuid(&open);
      if (!NT_SUCCESS(status) || !open.hAdapter)
         goto out;
      w->h_adapter = open.hAdapter;
   }
   if (!query_adapter_info(&w->dispatch, w->h_adapter, &adapter_info))
      goto out;

   /* 3. device（DXGK 每设备每 engine 单 context，与 dx11um 的 hDevice
    *    相互独立）。capset 查询要带 hDevice（参考客户端 UMD 同款用法）。 */
   stage = "create device";
   {
      D3DKMT_CREATEDEVICE create = {0};
      create.hAdapter = w->h_adapter;
      status = w->dispatch.CreateDevice(&create);
      if (!NT_SUCCESS(status) || !create.hDevice)
         goto out;
      w->h_device = create.hDevice;
   }

   /* 4. capset6：确认 host 是 msm，msm 参数装进 base.caps。mesa drm_hw.h 的
    *    capset 结构与 win3d 副本 prefix 一致，可整段拷贝（diff 只有追加）。 */
   stage = "DRM capset";
   memset(&w->base.caps, 0, sizeof(w->base.caps));
   if (!query_capset(w, w->h_device, VIRTIO_WDDM_CAPSET_ID_DRM, &w->base.caps,
                     sizeof(w->base.caps)))
      goto out;
   if (w->base.caps.context_type != VIRTGPU_DRM_CONTEXT_MSM)
      goto out;
   if (context_type != 0 && context_type != VIRTGPU_DRM_CONTEXT_MSM)
      goto out;

   /* 5. context（Graphics/3D engine） */
   stage = "create context";
   {
      D3DKMT_CREATECONTEXT create = {0};
      create.hDevice = w->h_device;
      create.NodeOrdinal = 0;          /* Graphics/3D engine */
      create.EngineAffinity = 1;
      create.ClientHint = D3DKMT_CLIENTHINT_VULKAN;
      /* win3d KMD 不读 CreateContext 私有数据（参数走 ContextInit escape，
       * lib.rs:1162-1163）。 */
      status = w->dispatch.CreateContext(&create);
      WDDM_DEBUG("CreateContext: status=0x%08lx\n", (ULONG)status);
      if (!NT_SUCCESS(status) || !create.hContext)
         goto out;
      w->h_context = create.hContext;
   }

   /* 6. ContextInit(capset=DRM)。KMD 会顺带建 shadow virgl（host 必须同时
    *    暴露 virgl capset，否则这里失败 —— kmd-contract 未确认 1）。 */
   stage = "context init";
   if (!context_init(w))
      goto out;

   /* Guest backing is optional: older KMDs reject this capability escape.
    * Cross-device sharing remains unsupported. */
   w->base.supports_cross_device = false;
   w->base.supports_guest_alloc = false;
   VIRTIO_WDDM_GuestAllocCaps guest_caps = {
      .tag = VIRTIO_WDDM_ESCAPE_GUEST_ALLOC_CAPS_TAG,
   };
   status = wddm_escape(w, w->h_device, 0, &guest_caps, sizeof(guest_caps));
   if (NT_SUCCESS(status) && guest_caps.supported && guest_caps.alignment == 65536)
      w->base.supports_guest_alloc = true;
   WDDM_DEBUG("guest allocation supported=%u alignment=%u\n",
              w->base.supports_guest_alloc, guest_caps.alignment);

   /* 7. ring（块 3）。必须在 ContextInit 之后：CREATE_BLOB 带 ctx_id，而
    *    EXEC_BUF 之前没 ContextInit 会得 REINITIALIZATION_NEEDED。 */
   stage = "shared ring";
   if (!init_shmem(w))
      goto out;

   ok = true;

out:
   if (!ok) {
      WDDM_DEBUG("connect failed at %s\n", stage);
      /* 统一走 wddm_close：它负责回收表里的分配（含可能已建出来的 ring）、
       * 销毁 table_lock、关句柄、卸 gdi32。 */
      wddm_close(&w->base);
      free(w);
      return NULL;
   }
   return &w->base;
}

/* 实现在 vdrm_virtgpu.c（Linux ioctl probe）；Windows 没有 guest pool
 * (CREATE_GUEST_HANDLE)，恒 false → 上层走系统堆大小。 */
bool
vdrm_guest_pool_stats(int fd, uint64_t *total, uint64_t *used,
                      uint64_t *largest_free)
{
   (void)fd; (void)total; (void)used; (void)largest_free;
   return false;
}
