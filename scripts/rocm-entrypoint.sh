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

# Ensure the app data directories are writable by the non-root user.
# /app/data/generations is normally a host bind-mount, so never chown/chmod it:
# adopt the uid/gid that owns it instead, the same way we adopt the GPU groups
# above. The rest of /app/data lives in the image or a named volume and is ours
# to fix, so it gets chowned to match (dirs only, no -R on user content).
gen=/app/data/generations
mkdir -p "$gen"
gen_uid=$(stat -c %u "$gen")
gen_gid=$(stat -c %g "$gen")
if [ "$gen_uid" = 0 ]; then
    # Docker created it for us; nothing on the host cares who owns it.
    chown voicebox:voicebox "$gen"
elif [ "$gen_uid" != "$(id -u voicebox)" ]; then
    getent group "$gen_gid" >/dev/null || groupadd -g "$gen_gid" hostdata
    usermod -u "$gen_uid" -g "$gen_gid" voicebox
    # Re-own what the old uid owned inside the image / named volume.
    find /app/data -path "$gen" -prune -o -exec chown "$gen_uid:$gen_gid" {} +
fi

for dir in /app/data/cache /app/data/profiles "$gen"; do
    mkdir -p "$dir"
    [ "$dir" = "$gen" ] || chown voicebox:voicebox "$dir"
    gosu voicebox test -w "$dir" || {
        echo "error: $dir is not writable by the voicebox user (uid $(id -u voicebox))" >&2
        [ "$dir" = "$gen" ] && echo "hint: make the host dir mounted there writable by that uid" >&2
        exit 1
    }
done

exec gosu voicebox "$@"
