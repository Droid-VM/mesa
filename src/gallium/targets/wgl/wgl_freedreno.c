/* SPDX-License-Identifier: MIT */
#include <windows.h>
#include "wgl_freedreno.h"
#include "pipe/p_screen.h"
#include "util/u_inlines.h"
#include "util/format/u_format.h"
#include "util/driconf.h"
#include "util/xmlconfig.h"
#include "freedreno/drm/freedreno_drm_public.h"

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

void
fd_wgl_present(struct pipe_context *ctx, struct pipe_resource *res, HDC hdc)
{
   struct pipe_transfer *transfer = NULL;
   const void *map = pipe_texture_map(ctx, res, 0, 0, PIPE_MAP_READ,
                                    0, 0, res->width0, res->height0, &transfer);
   if (!map)
      return;

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
   StretchDIBits(hdc, 0, 0, res->width0, res->height0,
                0, 0, res->width0, res->height0, map,
                (BITMAPINFO *)&bmi, DIB_RGB_COLORS, SRCCOPY);
   GdiFlush();
out:
   free(converted);
   pipe_texture_unmap(ctx, transfer);
}
