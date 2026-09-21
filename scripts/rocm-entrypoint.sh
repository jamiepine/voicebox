#!/bin/sh
set -e
# Join whatever groups own the mounted GPU nodes so /dev/kfd and /dev/dri work
# on any host (no RENDER_GID/VIDEO_GID needed), then drop to the app user.
for dev in /dev/kfd /dev/dri/render* /dev/nvidia*; do
    [ -e "$dev" ] || continue
    gid=$(stat -c %g "$dev")
    grp=$(getent group "$gid" | cut -d: -f1)
    [ -n "$grp" ] || {
        grp="gpu$gid"
        groupadd -g "$gid" "$grp"
    }
    usermod -aG "$grp" voicebox
done
# Ensure CUDA and NVML library symlinks and dynamic linker cache are available
if [ -e /usr/lib/x86_64-linux-gnu/libcuda.so.1 ] && [ ! -e /usr/lib/x86_64-linux-gnu/libcuda.so ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
fi
if [ -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 ] && [ ! -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so ]; then
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
fi
ldconfig 2>/dev/null || true

chown -R voicebox:voicebox /home/voicebox /app/data 2>/dev/null || true
exec gosu voicebox "$@"
