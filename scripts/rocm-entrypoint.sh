#!/bin/sh
set -e
# Join whatever groups own the mounted GPU nodes so /dev/kfd and /dev/dri work
# on any host (no RENDER_GID/VIDEO_GID needed), then drop to the app user.
for dev in /dev/kfd /dev/dri/render*; do
    [ -e "$dev" ] || continue
    gid=$(stat -c %g "$dev")
    grp=$(getent group "$gid" | cut -d: -f1)
    [ -n "$grp" ] || {
        grp="gpu$gid"
        groupadd -g "$gid" "$grp"
    }
    usermod -aG "$grp" voicebox
done

# Named volumes are created root-owned by Docker on first use; fix the
# mount points so the unprivileged voicebox user can write into them.
# Non-recursive: only the mount point itself needs fixing, not its
# contents, and the HF cache can be large.
chown voicebox:voicebox /app/data /app/data/generations /home/voicebox/.cache/huggingface 2>/dev/null || true

exec gosu voicebox "$@"
