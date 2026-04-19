"""
Force transformers.masking_utils to be bundled with its .py source alongside
the .pyc bytecode so the runtime hook in
backend/pyi_rth_torch_compiler_disable.py can source-patch it.

The patch forces the torch<2.6 code path, bypassing `with TransformGetItemToIndex()`
which our torch._dynamo no-op stub can't implement for real — the real context
manager uses dynamo graph transforms to avoid `.item()` calls inside vmap.
"""

module_collection_mode = "pyz+py"
