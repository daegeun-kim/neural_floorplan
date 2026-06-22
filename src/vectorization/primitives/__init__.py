"""Component primitive library for neural floorplan vectorization (v007/v008, run3)."""

from .base import BasePrimitive, ScaleInfo
from .door import DoorArcPrimitive, DoorLeafPrimitive, DoorOriginPrimitive
from .floor import FloorPrimitive
from .opening import OpeningPrimitive
from .scale import resolve_scale, snap_to_module_mm
from .wall import OuterWallLoopPrimitive, WallPrimitive
from .window import WindowPrimitive

__all__ = [
    "BasePrimitive",
    "ScaleInfo",
    "resolve_scale",
    "snap_to_module_mm",
    "FloorPrimitive",
    "WallPrimitive",
    "OuterWallLoopPrimitive",
    "OpeningPrimitive",
    "DoorOriginPrimitive",
    "DoorLeafPrimitive",
    "DoorArcPrimitive",
    "WindowPrimitive",
]
