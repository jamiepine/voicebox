#!/bin/sh
# Normalize ownership of the mounted data and model-cache directories, then drop
# privileges to the unprivileged "voicebox" user.
#
# The server runs as a non-root user for security, but Docker creates bind-mount
# host paths and freshly initialized named volumes owned by root. A non-root
# process then cannot write generated audio or the downloaded HuggingFace model
# cache. Running this entrypoint as root lets us chown the mount points at
# startup — regardless of host ownership — so it works out of the box for
# everyone. We then exec the server as voicebox via gosu.
set -e

if [ "$(id -u)" = "0" ]; then
    mkdir -p \
        /app/data/generations \
        /app/data/profiles \
        /app/data/cache \
        /home/voicebox/.cache/huggingface
    chown -R voicebox:voicebox /app/data /home/voicebox/.cache
    exec gosu voicebox "$@"
fi

# Already running as a non-root user (e.g. an explicit compose `user:` override):
# nothing to fix, just run the command.
exec "$@"
