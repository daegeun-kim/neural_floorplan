"""Tests for the cross-platform CairoSVG runtime loader."""

from __future__ import annotations

from types import SimpleNamespace

from src import cairo_runtime


def test_non_windows_uses_normal_import(monkeypatch):
    sentinel = SimpleNamespace()
    imported: list[str] = []

    monkeypatch.setattr(cairo_runtime.sys, "platform", "linux")
    monkeypatch.setattr(
        cairo_runtime.importlib,
        "import_module",
        lambda name: imported.append(name) or sentinel,
    )

    assert cairo_runtime.load_cairosvg() is sentinel
    assert imported == ["cairosvg"]


def test_windows_conda_aliases_cairo_dll_during_import(tmp_path, monkeypatch):
    cairo_dll = tmp_path / "Library" / "bin" / "cairo.dll"
    cairo_dll.parent.mkdir(parents=True)
    cairo_dll.touch()

    sentinel = SimpleNamespace()
    original_find_library = cairo_runtime.ctypes.util.find_library
    registered_directories: list[str] = []

    def fake_import(name: str):
        assert name == "cairosvg"
        assert cairo_runtime.ctypes.util.find_library("cairo") == str(cairo_dll)
        assert cairo_runtime.ctypes.util.find_library("cairo-2") == str(cairo_dll)
        assert cairo_runtime.ctypes.util.find_library("libcairo-2") == str(cairo_dll)
        return sentinel

    monkeypatch.setattr(cairo_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cairo_runtime.sys, "prefix", str(tmp_path))
    monkeypatch.delitem(cairo_runtime.sys.modules, "cairosvg", raising=False)
    monkeypatch.setattr(cairo_runtime.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        cairo_runtime.os,
        "add_dll_directory",
        lambda path: registered_directories.append(path) or SimpleNamespace(),
        raising=False,
    )
    monkeypatch.setattr(cairo_runtime, "_DLL_DIRECTORY_HANDLES", [])

    assert cairo_runtime.load_cairosvg() is sentinel
    assert registered_directories == [str(cairo_dll.parent)]
    assert cairo_runtime.ctypes.util.find_library is original_find_library


def test_windows_without_conda_cairo_preserves_normal_error_path(tmp_path, monkeypatch):
    sentinel = SimpleNamespace()
    imported: list[str] = []

    monkeypatch.setattr(cairo_runtime.sys, "platform", "win32")
    monkeypatch.setattr(cairo_runtime.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(
        cairo_runtime.importlib,
        "import_module",
        lambda name: imported.append(name) or sentinel,
    )

    assert cairo_runtime.load_cairosvg() is sentinel
    assert imported == ["cairosvg"]
