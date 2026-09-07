# Copyright 2023 Google LLC
# SPDX-License-Identifier: MIT

inc_virtio_vdrm = include_directories('.')

libvdrm_c_args = [no_override_init_args, soong_compat_c_args]
if with_amdgpu_virtio
  libvdrm_c_args += ['-DENABLE_DRM_AMDGPU']
endif
# Windows guest 没有 /dev/dri：后端换成 vdrm_wddm.c（D3DKMT →
# virtio_wddm_rs.sys）。virtgpu/vpipe 两个后端是 Linux 的 ioctl/socket
# 实现，在 Windows 上连头文件都编不过。
vdrm_files = ['vdrm.c', 'vdrm.h']
if host_machine.system() == 'windows'
  vdrm_files += ['vdrm_wddm.c']
else
  vdrm_files += ['vdrm_virtgpu.c', 'vdrm_vpipe.c']
endif

libvdrm = static_library(
  'libvdrm',
  vdrm_files,
  include_directories : [
    inc_include,
    inc_src,
    inc_virtio,
    inc_virtio_gpu,
  ],
  c_args : libvdrm_c_args,
  gnu_symbol_visibility: 'hidden',
  dependencies: [
    dep_libdrm,
    dep_valgrind,
    idep_mesautil,
  ],
  link_with: [
    _libmesa_util,
  ],
  build_by_default: false,
)
