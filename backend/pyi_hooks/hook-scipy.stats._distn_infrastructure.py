"""
Force scipy.stats._distn_infrastructure to be bundled with its .py source file
alongside the .pyc bytecode.

The runtime hook in backend/pyi_rth_torch_compiler_disable.py patches this
module's source at load time (the module has a `del obj` at line 369 that
raises NameError under PyInstaller's frozen importer). That patch reads the
source via loader.get_source(), which only works if the .py file was
actually collected into the bundle.
"""

module_collection_mode = "pyz+py"
