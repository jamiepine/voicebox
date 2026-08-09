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
# user can't write any cache (torch, spacy) outside the HF mount. Tolerate a
# read-only or root-squashed cache mount: the app only warns about that.
mkdir -p /home/voicebox/.cache/huggingface 2>/dev/null || true
chown voicebox:voicebox /home/voicebox/.cache /home/voicebox/.cache/huggingface 2>/dev/null || true

# Ensure the app data directories are writable by the non-root user.
# /app/data or /app/data/generations may be a host bind-mount, so never
# chown/chmod a host-owned one: adopt the uid/gid that owns it instead, the
# same way we adopt the GPU groups above (generated files then land on the
# host owned by the host user). Root-owned dirs were created by Docker or the
# image and are ours to fix, so they get chowned (dirs only, no -R on content).
data=/app/data
gen=$data/generations
host_uid=$(stat -c %u "$data")
host_gid=$(stat -c %g "$data")
if [ "$host_uid" = 0 ] || [ "$host_uid" = "$(id -u voicebox)" ]; then
    mkdir -p "$gen"
    host_uid=$(stat -c %u "$gen")
    host_gid=$(stat -c %g "$gen")
fi
if [ "$host_uid" != 0 ] && [ "$host_uid" != "$(id -u voicebox)" ]; then
    old_uid=$(id -u voicebox)
    getent group "$host_gid" >/dev/null || groupadd -g "$host_gid" hostdata
    usermod -u "$host_uid" -g "$host_gid" voicebox
    # Re-own what the old uid owned inside the image / named volume.
    find "$data" -uid "$old_uid" -exec chown "$host_uid:$host_gid" {} +
fi

for dir in "$data" "$data/cache" "$data/profiles" "$gen"; do
    mkdir -p "$dir" 2>/dev/null || true
    [ "$(stat -c %u "$dir")" = 0 ] && chown voicebox:voicebox "$dir" 2>/dev/null || true
    gosu voicebox test -w "$dir" || {
        echo "error: $dir is not writable by the voicebox user (uid $(id -u voicebox))" >&2
        echo "hint: make the host dir mounted there writable by that uid" >&2
        exit 1
    }
done

exec gosu voicebox "$@"
