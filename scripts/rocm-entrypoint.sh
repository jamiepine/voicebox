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

# Named volumes are created root-owned when the mount path is absent from the
# image, so the HuggingFace cache can end up unwritable by the app user.
# Create and hand it over before dropping privileges.
mkdir -p /home/voicebox/.cache/huggingface
chown -R voicebox:voicebox /home/voicebox/.cache

exec gosu voicebox "$@"
