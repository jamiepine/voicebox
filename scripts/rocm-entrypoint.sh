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

# Ensure the mounted data volume is writable by the non-root user.
# The Dockerfile chowns /app/data at build time, but a runtime volume mount
# re-creates it owned by root, so fix ownership here (still root) before
# dropping privileges.
chown -R voicebox:voicebox /app/data || true

exec gosu voicebox "$@"
