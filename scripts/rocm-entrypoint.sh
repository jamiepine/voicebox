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
# Docker creates the HF volume mountpoint's parent (~/.cache) as root on every
# container create, and a fresh volume is root-owned too — without this the app
# user can't write any cache (torch, spacy) outside the HF mount.
mkdir -p /home/voicebox/.cache/huggingface
chown voicebox:voicebox /home/voicebox/.cache /home/voicebox/.cache/huggingface

exec gosu voicebox "$@"
