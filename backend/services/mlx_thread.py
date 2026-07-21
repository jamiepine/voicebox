"""Single dedicated worker thread for all MLX GPU work.

MLX's Metal command encoder/stream is thread-local: it binds to whichever
thread first touches the GPU device. ``asyncio.to_thread()`` uses the event
loop's default executor, which hands successive calls to different worker
threads — a model loaded on one thread and generated on another raises
"There is no Stream(gpu, N) in current thread" (issue #699).

Routing every MLX load, generate, transcribe and unload through this one
worker keeps them on a single thread. Because the pool has a single worker,
submitted jobs also run to completion one at a time in submission order, so a
load-then-infer pair submitted as one job cannot be interleaved with an unload
or a different-size load from another request.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-worker")


def run_on_mlx_thread(func, *args):
    """Run ``func(*args)`` on the single dedicated MLX worker thread."""
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(_mlx_executor, func, *args)


def clear_mlx_cache() -> None:
    """Return MLX's cached unified memory to the OS after a model is freed.

    Must run on the MLX worker thread (call it from an unload that is already
    routed through ``run_on_mlx_thread``). ``clear_cache`` moved out of the
    ``mlx.core.metal`` namespace in newer MLX, so resolve it from either.
    """
    import mlx.core as mx

    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear is not None:
        clear()
