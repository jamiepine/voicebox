"""Single dedicated worker thread for all MLX model operations.

MLX GPU streams are thread-local, and some MLX libraries bind a stream to
the thread that first imports them — most notably ``mlx_lm.generate``, which
runs ``generation_stream = mx.new_stream(...)`` at module import. In a
threaded ASGI server the default ``asyncio.to_thread`` pool hands successive
model calls to *different* worker threads, so a stream created on one thread
is missing on another. That surfaces as ``There is no Stream(gpu, N) in
current thread`` and, under worse timing, a hard SIGABRT inside
``mlx::core::metal::get_command_encoder``.

Routing every MLX load / inference / unload through ONE worker thread keeps
stream affinity intact and serializes Metal command encoding (MLX is not
safe under concurrent GPU eval from multiple threads). Import-time streams
are created on this thread the first time a model loads, and every later
call runs on the same thread, so they always resolve.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

# max_workers=1 is load-bearing: it is what guarantees a single, stable OS
# thread for all MLX work.
_MLX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


async def run_mlx(fn, *args, **kwargs):
    """Await ``fn(*args, **kwargs)`` on the dedicated MLX thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _MLX_EXECUTOR, functools.partial(fn, *args, **kwargs)
    )
