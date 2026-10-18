/* SPDX-License-Identifier: MIT */
#ifndef WGL_FREEDRENO_H
#define WGL_FREEDRENO_H
#include <windows.h>
struct pipe_screen;
struct pipe_context;
struct pipe_resource;
struct stw_shared_surface;
struct pipe_screen *fd_wgl_create_screen(void);
void fd_wgl_present(struct pipe_context *ctx, struct pipe_resource *res, HDC hdc);

/* stw_winsys::compose (see stw_winsys.h).  Composes `res` into the KMD
 * shared surface `dest`, falling back to the GDI present of the window DC
 * recorded by fd_wgl_present() when `dest` cannot be used as a GPU
 * resource, so the screen is never left stale. */
void fd_wgl_compose(struct pipe_screen *screen, struct pipe_resource *res,
                    struct stw_shared_surface *dest, LPCRECT pRect,
                    ULONGLONG PresentHistoryToken);
#endif
