"""Load CairoSVG with the active Conda environment's Cairo DLL on Windows.

Conda-forge installs Cairo as ``Library/bin/cairo.dll``. CairoCFFI's Windows
fallback names do not include that filename, so a normal ``import cairosvg`` can
fail even when the documented native dependency is installed. This module
bridges that naming difference without copying or modifying environment files.
"""

from __future__ import annotations

import ctypes.util
import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

_CAIRO_LIBRARY_NAMES = frozenset({"cairo", "cairo-2", "libcairo-2"})
_DLL_DIRECTORY_HANDLES: list[object] = []


def _conda_cairo_dll() -> Path | None:
    """Return Conda's Cairo DLL when running in a compatible Windows environment."""
    if sys.platform != "win32":
        return None

    candidate = Path(sys.prefix) / "Library" / "bin" / "cairo.dll"
    return candidate if candidate.is_file() else None


def load_cairosvg() -> ModuleType:
    """Import CairoSVG, resolving Conda's Cairo DLL name on Windows when needed."""
    cairo_dll = _conda_cairo_dll()
    if cairo_dll is None or "cairosvg" in sys.modules:
        return importlib.import_module("cairosvg")

    dll_directory = cairo_dll.parent
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None and not _DLL_DIRECTORY_HANDLES:
        # Keep the handle alive for the process lifetime so Cairo's dependent
        # DLLs remain discoverable after this function returns.
        _DLL_DIRECTORY_HANDLES.append(add_dll_directory(str(dll_directory)))

    original_find_library = ctypes.util.find_library

    def find_library(name: str) -> str | None:
        if name in _CAIRO_LIBRARY_NAMES:
            return str(cairo_dll)
        return original_find_library(name)

    # CairoCFFI imports find_library by value while it initializes. Restore the
    # global function immediately afterward; CairoCFFI retains the loaded handle.
    ctypes.util.find_library = find_library
    try:
        return importlib.import_module("cairosvg")
    finally:
        ctypes.util.find_library = original_find_library
