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
# Docker creates /home/voicebox/.cache as root on every container start, and
# named volumes mounted under it are also root-owned. The app runs as uid 999
# (voicebox), so cache writes (torch hub, spacy, etc.) fail with Permission
# denied. chown the parent and the mountpoint explicitly: chown on the parent
# does not propagate into volume content.
mkdir -p /home/voicebox/.cache/huggingface
chown -R voicebox:voicebox /home/voicebox/.cache /home/voicebox/.cache/huggingface

exec gosu voicebox "$@"
