/* SPDX-License-Identifier: MIT */
#ifndef WGL_FREEDRENO_H
#define WGL_FREEDRENO_H
#include <windows.h>
struct pipe_screen;
struct pipe_context;
struct pipe_resource;
struct pipe_screen *fd_wgl_create_screen(void);
void fd_wgl_present(struct pipe_context *ctx, struct pipe_resource *res, HDC hdc);
#endif
