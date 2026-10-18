/* SPDX-License-Identifier: MIT */
#include <windows.h>
#include <stdlib.h>
#include <string.h>
#include "wgl_freedreno.h"
#include "stw_context.h"
#include "stw_st.h"
#include "pipe/p_screen.h"
#include "util/u_inlines.h"
#include "util/format/u_format.h"
#include "util/driconf.h"
#include "util/xmlconfig.h"
#include "freedreno/drm/freedreno_drm_public.h"
#include "util/simple_mtx.h"
#include "util/os_time.h"
#include "util/log.h"

struct pipe_screen *
fd_wgl_create_screen(void)
{
   static const driOptionDescription options[] = {
#include "freedreno/driinfo_freedreno.h"
   };
   driOptionCache info = {0}, cache = {0};
   driParseOptionInfo(&info, options, ARRAY_SIZE(options));
   struct pipe_screen_config config = {.options = &cache, .options_info = &info};
   struct pipe_screen *screen = fd_drm_screen_create_renderonly(-1, NULL, &config);
   driDestroyOptionCache(&cache);
   driDestroyOptionInfo(&info);
   return screen;
}

/* Context of the latest present.  stw_winsys::compose has no HDC argument,
 * but the GDI fallback below needs one.  fd_wgl_present() runs immediately
 * before it, inside the same framebuffer-locked present call on the same
 * thread. */
static HDC fd_wgl_compose_hdc;

/* ---------------------------------------------------------------------------
 * droidvm-staging2 (2026-10-09; E53 revision): persistent staging readback.
 *
 * fd_wgl_present() needs the WGL colour buffer's pixels in CPU memory for the
 * GDI present.  That colour buffer is TILED, so fd_resource_transfer_map()
 * takes the resource_transfer_map_staging() path, which for EVERY frame:
 *   1. fd_alloc_staging()   -> virtio_bo_new -> wddm_bo_create (host GEM_NEW
 *                              ccmd + D3DKMTCreateAllocation + ResourceInfo
 *                              escape + residency)
 *   2. fd_blit_to_staging() -> full-frame GPU blit (tiled -> linear)
 *   3. fd_resource_wait()   -> waits for THAT blit (KMD ResourceBusy escape +
 *                              host GEM_CPU_PREP ccmd, polled while EBUSY)
 *   4. fd_bo_map()          -> KMD BlobMap escape (build MDL, map the pages)
 *   5. transfer_unmap()     -> BlobMap UNMAP escape + DestroyAllocation + host
 *                              gem close
 * Measured with the ICD probe: PRES_A = 1555 us per frame = 69% of the whole
 * ICD present.
 *
 * staging2 keeps TWO persistent linear staging buffers and rotates them per
 * *present* (not per drawable): present N blits the colour buffer into
 * buf[next] and hands the CPU the buffer filled by present N-1, so
 *   - steps 1, 4 and 5 happen once per process instead of once per frame;
 *   - step 3 lands on GPU work submitted a whole present ago, so the render
 *     thread no longer stalls on the blit;
 *   - fd's own linear map path still does the synchronisation
 *     (pending()/resource_busy() -> fd_resource_wait(), see
 *     freedreno_resource.c), so the pixels handed out are guaranteed complete
 *     -- they are simply one present old;
 *   - the colour buffer keeps its TILED layout.  Forcing it linear (E51)
 *     measured slower overall than the readback it removed.
 * Cost: the presented image lags one present.  Set DROIDVM_STAGING2=reuse to
 * keep the persistent pair but read back this present, or
 * DROIDVM_STAGING2=off for the stock fd path (bisect without a rebuild).
 *
 * E53 -- why the cache key must NOT contain the presented pipe_resource:
 * stw_st_swap_framebuffer_locked() swaps textures[FRONT_LEFT] and
 * textures[BACK_LEFT] after every successful present (stw_st.c:611-614; the
 * flush_front path swaps the same pair at stw_st.c:533-537), so a
 * double-buffered client presents a *different* pipe_resource every frame:
 * A, B, A, B, ...  The first staging2 revision compared `owner == res`, which
 * therefore never held for two frames in a row: both staging BOs were
 * destroyed and recreated on every single frame -- exactly the 1555 us
 * readback this code exists to remove, plus one stderr line per frame.
 * The pair is now keyed only on what genuinely requires new buffers (owning
 * screen, format, width, height); the deferred readback is driven by the
 * present generation stamped into each buffer, so an alternating swap chain
 * gets the same one-present lag as a single-resource client instead of frame
 * doubling.  Trade-off, stated plainly: the pair is shared by all drawables of
 * one geometry, so a second same-size window would see the first window's
 * capture for one present.  Keying on the drawable pointer again would bring
 * the per-frame rebuild back (E53), and keying on hdc is not safe either --
 * stw_st.c:524-531 documents that HDCs may be invalidated by the application.
 * Safety: on the first staged frame the capture is compared against fd's own
 * staging path (fd_wgl_stage2_selfcheck); any mismatch disables staging2 for
 * the process instead of presenting wrong pixels.  A pair that has to be
 * rebuilt over and over (the E53 signature) disables staging2 too, so a future
 * key regression degrades to the stock path instead of stuttering forever.
 * ------------------------------------------------------------------------- */

#define FD_STAGING2_BUFS 2

/* Startup thrash guard: this many rebuilds while more than half of the staged
 * frames needed one means the cache key is unstable (E53).  Give up on
 * staging2 -- the stock fd path is the known-smooth present -- rather than
 * rebuild BOs and write stderr forever. */
#define FD_STAGING2_THRASH_REBUILDS 16

/* Distinct presented resources tracked for the summary line only. */
#define FD_STAGING2_SEEN 4

struct fd_wgl_stage2 {
   /* Cache key: the only things that actually require new staging buffers.
    * screen is compared, never dereferenced.  See the E53 note above for why
    * the presented resource is not part of it. */
   struct pipe_screen *screen;
   enum pipe_format format;
   unsigned width, height;

   struct pipe_resource *buf[FD_STAGING2_BUFS];
   /* Present generation that filled each buffer (0 = never filled); the
    * buffer stamped with the previous generation is the one handed out. */
   uint64_t fill_gen[FD_STAGING2_BUFS];
   unsigned next;                 /* blit target for the next present */

   /* Diagnostics only -- never part of the key. */
   struct pipe_context *ctx;
   struct pipe_resource *last_res;
};

static struct fd_wgl_stage2 fd_wgl_stage2;
static simple_mtx_t fd_wgl_stage2_lock = SIMPLE_MTX_INITIALIZER;

/* Counters for the periodic summary line below (evidence, not a probe). */
static unsigned s2_frames, s2_deferred, s2_rebuilds, s2_fallbacks;
static int64_t s2_blit_us, s2_map_us;
static uint64_t s2_generation;   /* staged presents so far */
static bool s2_verified;    /* self-check already run (once per process) */
static bool s2_disabled;    /* self-check/thrash guard failed: stay on E48 */
static struct pipe_resource *s2_seen[FD_STAGING2_SEEN];
static unsigned s2_n_seen;

static int
fd_wgl_staging2_mode(void)
{
   static int mode = -1;

   if (s2_disabled)
      return 0;

   if (mode < 0) {
      const char *e = getenv("DROIDVM_STAGING2");

      if (e && (!strcmp(e, "off") || !strcmp(e, "0")))
         mode = 0;              /* stock fd staging: one fresh BO per frame */
      else if (e && !strcmp(e, "reuse"))
         mode = 1;              /* persistent pair, read back this frame */
      else
         mode = 2;              /* persistent pair, deferred by one present */

      mesa_loge("droidvm-staging2: mode=%d (%s)", mode,
                mode == 0 ? "off, stock fd staging" :
                mode == 1 ? "reuse, same-frame readback" :
                            "reuse, one-present-deferred readback");
   }

   return mode;
}

static void
fd_wgl_stage2_drop(void)
{
   pipe_resource_reference(&fd_wgl_stage2.buf[1], NULL);
   pipe_resource_reference(&fd_wgl_stage2.buf[0], NULL);
   fd_wgl_stage2.screen = NULL;
   fd_wgl_stage2.format = PIPE_FORMAT_NONE;
   fd_wgl_stage2.width = fd_wgl_stage2.height = 0;
   fd_wgl_stage2.fill_gen[0] = fd_wgl_stage2.fill_gen[1] = 0;
   fd_wgl_stage2.next = 0;
   fd_wgl_stage2.ctx = NULL;
   fd_wgl_stage2.last_res = NULL;
}

/* (Re)create the pair for this geometry.  Returns false when the buffers could
 * not be created or when the thrash guard gave up; the caller then uses the
 * stock fd path (never a black screen: that fallback is exactly what shipped
 * in E48). */
static bool
fd_wgl_stage2_build(struct pipe_context *ctx, struct pipe_resource *res)
{
   struct pipe_resource tmpl;
   struct pipe_screen *screen = res->screen;
   unsigned i;

   fd_wgl_stage2_drop();

   memset(&tmpl, 0, sizeof(tmpl));
   tmpl.target = PIPE_TEXTURE_2D;
   tmpl.format = res->format;
   tmpl.width0 = res->width0;
   tmpl.height0 = res->height0;
   tmpl.depth0 = 1;
   tmpl.array_size = 1;
   tmpl.nr_samples = res->nr_samples;
   tmpl.last_level = 0;
   tmpl.usage = PIPE_USAGE_STAGING;
   /* PIPE_BIND_LINEAR: get_best_layout() -> FD_LAYOUT_LINEAR, so
    * rsc->layout.tile_mode == 0 and the BO is not FD_BO_NOMAP => it stays a
    * mappable guest blob and map() is that BO's own CPU mapping instead of a
    * second, nested staging blit.
    * PIPE_BIND_RENDER_TARGET: this buffer is the blit destination.
    * PIPE_USAGE_STAGING + PIPE_RESOURCE_FLAG_MAP_COHERENT make fd request
    * FD_BO_CACHED_COHERENT (freedreno_resource.c, realloc_bo()), which on
    * Windows guest allocations becomes MSM_BO_CACHED_COHERENT and is mapped
    * MmCached by the KMD guest-backing path -- i.e. NOT the ~750 MB/s
    * uncached Aperture3D mapping the author measured. */
   tmpl.bind = PIPE_BIND_LINEAR | PIPE_BIND_RENDER_TARGET |
               PIPE_BIND_SAMPLER_VIEW;
   tmpl.flags = PIPE_RESOURCE_FLAG_MAP_COHERENT;

   for (i = 0; i < FD_STAGING2_BUFS; i++) {
      fd_wgl_stage2.buf[i] = screen->resource_create(screen, &tmpl);
      if (!fd_wgl_stage2.buf[i]) {
         mesa_loge("droidvm-staging2: staging buffer %u of %ux%u %s could not "
                   "be created -- falling back to fd's per-frame staging",
                   i, res->width0, res->height0, util_format_name(res->format));
         fd_wgl_stage2_drop();
         return false;
      }
   }

   fd_wgl_stage2.screen = screen;
   fd_wgl_stage2.format = res->format;
   fd_wgl_stage2.width = res->width0;
   fd_wgl_stage2.height = res->height0;
   fd_wgl_stage2.next = 0;
   fd_wgl_stage2.ctx = ctx;
   fd_wgl_stage2.last_res = res;
   s2_rebuilds++;

   mesa_loge("droidvm-staging2: staging pair %ux%u %s -> %p / %p (rebuild #%u, "
             "staged frames=%u, drawable=%p)",
             res->width0, res->height0, util_format_name(res->format),
             (void *)fd_wgl_stage2.buf[0], (void *)fd_wgl_stage2.buf[1],
             s2_rebuilds, s2_frames, (void *)res);

   /* E53 guard: rebuilt again and again while staging frames too -- the key is
    * unstable, so stop before this becomes the per-frame BO churn it replaces. */
   if (s2_rebuilds > FD_STAGING2_THRASH_REBUILDS && s2_rebuilds > s2_frames / 2) {
      s2_disabled = true;
      mesa_loge("droidvm-staging2: %u pair rebuilds after only %u staged frames "
                "-- the cache key is thrashing; disabling droidvm-staging2 for "
                "this process (present falls back to the stock fd staging path)",
                s2_rebuilds, s2_frames);
      fd_wgl_stage2_drop();
      return false;
   }

   return true;
}

static bool
fd_wgl_stage2_usable(struct pipe_context *ctx, struct pipe_resource *res)
{
   if (fd_wgl_stage2.buf[0] && fd_wgl_stage2.buf[1] &&
       fd_wgl_stage2.screen == res->screen &&
       fd_wgl_stage2.format == res->format &&
       fd_wgl_stage2.width == res->width0 &&
       fd_wgl_stage2.height == res->height0) {
      fd_wgl_stage2.ctx = ctx;         /* diagnostics only */
      fd_wgl_stage2.last_res = res;    /* diagnostics only */
      return true;
   }

   return fd_wgl_stage2_build(ctx, res);
}

/* Capture the drawable into one of our staging buffers. */
static void
fd_wgl_stage2_capture(struct pipe_context *ctx, struct pipe_resource *dst,
                      struct pipe_resource *src)
{
   struct pipe_blit_info blit;

   memset(&blit, 0, sizeof(blit));
   blit.src.resource = src;
   blit.src.level = 0;
   blit.src.format = src->format;
   blit.src.box.width = src->width0;
   blit.src.box.height = src->height0;
   blit.src.box.depth = 1;
   blit.dst.resource = dst;
   blit.dst.level = 0;
   blit.dst.format = dst->format;
   blit.dst.box.width = dst->width0;
   blit.dst.box.height = dst->height0;
   blit.dst.box.depth = 1;
   blit.mask = util_format_get_mask(dst->format);
   blit.filter = PIPE_TEX_FILTER_NEAREST;
   blit.scissor_enable = false;

   ctx->blit(ctx, &blit);
}

/* One-time proof that the new capture is the same picture as the stock fd
 * staging path (E48) it replaces.  pipe_context::blit() returns void, so an
 * unsupported blit would fail silently and present garbage; instead of hoping,
 * compare the two captures here.  For 4-byte formats only the R,G,B bytes are
 * compared: the GDI DIB present in this file ignores byte 3 (X or A), and it is
 * the only channel the blit's mask leaves undefined.  Returns false only on a
 * real mismatch. */
static bool
fd_wgl_stage2_selfcheck(struct pipe_context *ctx, struct pipe_resource *res,
                        const void *ours, unsigned our_stride)
{
   struct pipe_transfer *vt = NULL;
   const uint8_t *theirs;
   const bool rgb_only = util_format_get_blocksize(res->format) == 4;
   const unsigned row_bytes = util_format_get_stride(res->format, res->width0);
   unsigned row, bad = 0;

   if (util_format_get_blockwidth(res->format) != 1 ||
       util_format_get_blockheight(res->format) != 1) {
      mesa_loge("droidvm-staging2: self-check skipped for block format %s -- "
                "keeping the new path (unverified)", util_format_name(res->format));
      return true;
   }

   theirs = pipe_texture_map(ctx, res, 0, 0, PIPE_MAP_READ, 0, 0,
                             res->width0, res->height0, &vt);
   if (!theirs || !vt || our_stride < row_bytes || vt->stride < row_bytes) {
      mesa_loge("droidvm-staging2: self-check could not run (stock map %p "
                "stride %u, ours %u, row %u) -- keeping the new path (unverified)",
                (const void *)theirs, vt ? vt->stride : 0, our_stride, row_bytes);
      if (vt)
         pipe_texture_unmap(ctx, vt);
      return true;
   }

   for (row = 0; row < res->height0 && !bad; row++) {
      const uint8_t *a = (const uint8_t *)ours + (size_t)row * our_stride;
      const uint8_t *b = theirs + (size_t)row * vt->stride;

      if (!rgb_only) {
         if (memcmp(a, b, row_bytes))
            bad = row + 1;
      } else {
         unsigned x;

         for (x = 0; x < res->width0; x++) {
            if (a[x * 4 + 0] != b[x * 4 + 0] ||
                a[x * 4 + 1] != b[x * 4 + 1] ||
                a[x * 4 + 2] != b[x * 4 + 2]) {
               bad = row + 1;
               break;
            }
         }
      }
   }

   pipe_texture_unmap(ctx, vt);

   if (bad) {
      s2_disabled = true;
      mesa_loge("droidvm-staging2: SELF-CHECK FAILED at row %u of %ux%u %s -- "
                "the staging2 capture does not match fd's own staging path; "
                "disabling droidvm-staging2 for this process (present continues "
                "on the E48 path)", bad, res->width0, res->height0,
                util_format_name(res->format));
      return false;
   }

   mesa_loge("droidvm-staging2: self-check OK -- %ux%u %s capture is identical "
             "to fd's staging path over all %u rows%s",
             res->width0, res->height0, util_format_name(res->format),
             res->height0, rgb_only ? " (R,G,B compared)" : "");
   return true;
}

/* Map the staging copy.  Called with fd_wgl_stage2_lock held; returns NULL
 * when the caller must use the stock fd path. */
static void *
fd_wgl_stage2_map(struct pipe_context *ctx, struct pipe_resource *res,
                  struct pipe_transfer **transfer, const char **path,
                  int64_t *blit_us)
{
   const int mode = fd_wgl_staging2_mode();
   uint64_t gen;
   unsigned wi, ri, s;
   bool deferred;
   int64_t t0, t1;
   void *map;

   if (!fd_wgl_stage2_usable(ctx, res))
      return NULL;

   wi = fd_wgl_stage2.next;
   ri = wi ^ 1;
   /* The buffer stamped with the previous present's generation is the one to
    * hand out: its blit was submitted a whole present ago, so the wait below
    * does not stall on the GPU.  mode 1 ("reuse") reads back this present. */
   gen = ++s2_generation;
   deferred = (mode >= 2) && fd_wgl_stage2.fill_gen[ri] != 0 &&
              fd_wgl_stage2.fill_gen[ri] == gen - 1;

   t0 = os_time_get_nano();
   fd_wgl_stage2_capture(ctx, fd_wgl_stage2.buf[wi], res);
   t1 = os_time_get_nano();
   *blit_us = (t1 - t0) / 1000;

   fd_wgl_stage2.fill_gen[wi] = gen;

   /* Startup, the frame right after a rebuild, and any present whose
    * predecessor did not fill its buffer (a failed map) read back what was
    * just captured instead of an older buffer. */
   if (!deferred)
      ri = wi;

   *transfer = NULL;
   map = pipe_texture_map(ctx, fd_wgl_stage2.buf[ri], 0, 0, PIPE_MAP_READ,
                          0, 0, res->width0, res->height0, transfer);
   if (!map) {
      s2_fallbacks++;
      return NULL;
   }

   /* First staged frame: verify the capture against fd's own staging path
    * before presenting anything out of it. */
   if (!deferred && !s2_verified) {
      s2_verified = true;
      if (!fd_wgl_stage2_selfcheck(ctx, res, map, (*transfer)->stride)) {
         pipe_texture_unmap(ctx, *transfer);
         *transfer = NULL;
         s2_fallbacks++;
         return NULL;
      }
   }

   if (mode >= 2)
      fd_wgl_stage2.next = wi ^ 1;

   *path = deferred ? "staging2(deferred, prev present)" : "staging2(sync)";

   /* Diagnostics only: how many distinct drawables go through this one pair.
    * 2 is the normal double-buffered swap chain that caused E53. */
   for (s = 0; s < s2_n_seen; s++) {
      if (s2_seen[s] == res)
         break;
   }
   if (s == s2_n_seen && s2_n_seen < FD_STAGING2_SEEN)
      s2_seen[s2_n_seen++] = res;

   s2_frames++;
   if (deferred)
      s2_deferred++;
   s2_blit_us += *blit_us;
   s2_map_us += (os_time_get_nano() - t1) / 1000;

   if ((s2_frames % 600) == 0) {
      mesa_loge("droidvm-staging2: frames=%u deferred=%u(%.0f%%) rebuilds=%u "
                "fallbacks=%u drawables=%u avg blit=%lldus avg map+wait=%lldus",
                s2_frames, s2_deferred,
                100.0 * s2_deferred / (double)s2_frames, s2_rebuilds,
                s2_fallbacks, s2_n_seen,
                (long long)(s2_blit_us / (int64_t)s2_frames),
                (long long)(s2_map_us / (int64_t)s2_frames));
   }

   return map;
}

static void
fd_wgl_present_resource(struct pipe_context *ctx, struct pipe_resource *res,
                        HDC hdc)
{
   struct pipe_transfer *transfer = NULL;
   int64_t _t0, _t1, _blit_us = 0;
   const void *map = NULL;
   const char *path = "fd-staging(fresh BO per frame)";
   bool locked = false;

   /* Same tolerance as the P3 branch, which also passes a possibly-NULL
    * pipe down to fd_wgl_present(). */
   if (!ctx || !res || !hdc)
      return;

   _t0 = os_time_get_nano();

   /* droidvm-staging2: capture into the persistent staging pair and hand out a
    * buffer whose blit is already complete.  The lock is held for the whole
    * present because the CPU reads that buffer through StretchDIBits below;
    * stw already serialises presents per drawable, so the uncontended case
    * costs nothing. */
   if (fd_wgl_staging2_mode() && res->target == PIPE_TEXTURE_2D) {
      simple_mtx_lock(&fd_wgl_stage2_lock);
      locked = true;
      map = fd_wgl_stage2_map(ctx, res, &transfer, &path, &_blit_us);
   }

   if (!map) {
      /* Fallback (mode=off, unusable drawable, allocation failure, a failed
       * self-check, or a map failure): the stock fd path, which is what
       * shipped in E48 -- a fresh staging BO per frame for the tiled colour
       * buffer. */
      transfer = NULL;
      map = pipe_texture_map(ctx, res, 0, 0, PIPE_MAP_READ,
                             0, 0, res->width0, res->height0, &transfer);
   }

   _t1 = os_time_get_nano();
   {
      static bool _logged = false;
      if (!_logged) {
         _logged = true;
         mesa_loge("fd_wgl_present: format=%s %ux%u readback=%lld us "
                   "blit=%lld us path=%s stride=%u map=%p",
                   util_format_name(res->format), res->width0, res->height0,
                   (long long)((_t1 - _t0) / 1000), (long long)_blit_us, path,
                   transfer ? transfer->stride : 0, map);
      }
   }
   if (!map) {
      if (locked)
         simple_mtx_unlock(&fd_wgl_stage2_lock);
      return;
   }

   BITMAPV5HEADER bmi = {0};
   bmi.bV5Size = sizeof(bmi);
   bmi.bV5Height = -(LONG)res->height0;
   bmi.bV5Planes = 1;
   bmi.bV5BitCount = 32;
   bmi.bV5Compression = BI_RGB;
   unsigned stride = transfer->stride;
   void *converted = NULL;
   if (res->format == PIPE_FORMAT_R8G8B8A8_UNORM ||
       res->format == PIPE_FORMAT_R8G8B8X8_UNORM ||
       res->format == PIPE_FORMAT_R8G8B8A8_SRGB ||
       res->format == PIPE_FORMAT_R8G8B8X8_SRGB) {
      bmi.bV5Compression = BI_BITFIELDS;
      bmi.bV5RedMask = 0xff;
      bmi.bV5GreenMask = 0xff00;
      bmi.bV5BlueMask = 0xff0000;
   } else if (res->format != PIPE_FORMAT_B8G8R8A8_UNORM &&
              res->format != PIPE_FORMAT_B8G8R8X8_UNORM &&
              res->format != PIPE_FORMAT_B8G8R8A8_SRGB &&
              res->format != PIPE_FORMAT_B8G8R8X8_SRGB) {
      stride = res->width0 * 4;
      converted = malloc((size_t)stride * res->height0);
      if (!converted || !util_format_translate(PIPE_FORMAT_B8G8R8X8_UNORM,
             converted, stride, 0, 0, res->format, map, transfer->stride,
             0, 0, res->width0, res->height0))
         goto out;
      map = converted;
   }
   /* DIB width includes row padding; the source rectangle excludes it. */
   bmi.bV5Width = stride / 4;
   {
      static bool _logged2 = false;
      if (!_logged2) {
         _logged2 = true;
         mesa_loge("fd_wgl_present: convert_path=%s converted=%p stride=%u dib_width=%d",
                   converted ? "yes(malloc+util_format_translate)" : "no",
                   converted, stride, (int)bmi.bV5Width);
      }
   }
   StretchDIBits(hdc, 0, 0, res->width0, res->height0,
                0, 0, res->width0, res->height0, map,
                (BITMAPINFO *)&bmi, DIB_RGB_COLORS, SRCCOPY);
   /* presentfast: per-frame GdiFlush removed (StretchDIBits is synchronous); restore if artifacts */
out:
   free(converted);
   pipe_texture_unmap(ctx, transfer);
   if (locked)
      simple_mtx_unlock(&fd_wgl_stage2_lock);
}

void
fd_wgl_present(struct pipe_context *ctx, struct pipe_resource *res, HDC hdc)
{
   fd_wgl_compose_hdc = hdc;
   fd_wgl_present_resource(ctx, res, hdc);
}

/*
 * stw_winsys::compose -- see stw_winsys.h:119 ("Blit the color buffer into a
 * shared surface").
 *
 * Contract: on return the frame must be on screen.  This ICD can only reach
 * the window through GDI (the WDDM/DWM redirection surface), so unless `dest`
 * can be turned into a pipe_resource the only correct action is the GDI
 * present -- i.e. exactly what the P3 branch does.  Turning `dest` into a
 * pipe_resource needs the KMD's shared-surface contract (allocation handle +
 * private driver data), which the virtio WDDM KMD does not expose to this
 * ICD today (see the E47 note in the bug report); `dst_res` below is
 * therefore always NULL and the fallback always runs.
 *
 * Why not present nothing and fall back to P3 instead?  Because `compose`
 * returns void and stw_framebuffer.c cannot tell success from failure: a
 * silent no-op here would leave the window stale (black/frozen screen).  See
 * the E47 recommendation -- the winsys needs a status return, or the compose
 * attempt must move into the winsys `present` entry point.
 */
void
fd_wgl_compose(struct pipe_screen *screen, struct pipe_resource *res,
               struct stw_shared_surface *dest, LPCRECT pRect,
               ULONGLONG PresentHistoryToken)
{
   struct pipe_context *pipe = stw_st_get_pipe();
   struct pipe_resource *dst_res = NULL;   /* not openable yet, see above */
   int w = pRect ? (pRect->right - pRect->left) : (int)res->width0;
   int h = pRect ? (pRect->bottom - pRect->top) : (int)res->height0;
   static unsigned n_compose;

   n_compose++;

   if (dst_res && pipe && screen && w > 0 && h > 0 &&
       res->width0 == (unsigned)w && res->height0 == (unsigned)h) {
      struct pipe_blit_info blit;

      mesa_loge_once("fd_wgl_compose: blitting %ux%u into shared surface %p "
                     "on the GPU", res->width0, res->height0, (void *)dest);

      memset(&blit, 0, sizeof blit);
      blit.src.resource = res;
      blit.src.level = 0;
      blit.src.format = res->format;
      blit.src.box.width = w;
      blit.src.box.height = h;
      blit.src.box.depth = 1;
      blit.dst.resource = dst_res;
      blit.dst.level = 0;
      blit.dst.format = dst_res->format;
      blit.dst.box.width = w;
      blit.dst.box.height = h;
      blit.dst.box.depth = 1;
      blit.mask = PIPE_MASK_RGBA;
      blit.filter = PIPE_TEX_FILTER_NEAREST;
      blit.scissor_enable = false;
      pipe->blit(pipe, &blit);
      pipe->flush(pipe, NULL, 0);
      (void)PresentHistoryToken;
      return;
   }

   mesa_loge_once("fd_wgl_compose: call #%u dest=%p is not usable as a GPU "
                  "resource (no KMD shared-surface contract); falling back to "
                  "the GDI readback present", n_compose, (void *)dest);

   /* Correctness first: present exactly the way the P3 branch does
    * (stw_framebuffer.c -> stw_winsys->present -> fd_wgl_present).  pipe may
    * be NULL here -- the P3 branch passes a possibly-NULL pipe down to
    * fd_wgl_present as well. */
   if (fd_wgl_compose_hdc)
      fd_wgl_present_resource(pipe, res, fd_wgl_compose_hdc);
}
