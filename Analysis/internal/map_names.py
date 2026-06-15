from __future__ import annotations

MAP_DISPLAY_NAMES: dict[str, str] = {
    "gangjin.full": "Highway (Full-scale site map)",
    "gangjin.up": "Highway (Sub-scale: upper region)",
    "gangjin.down": "Highway (Sub-scale: lower region)",
    "seocho.full": "Office Building (Full-scale site map)",
    "seocho.up": "Office Building (Sub-scale: upper region)",
    "seocho.down": "Office Building (Sub-scale: lower region)",
    "sejong.full": "Industrial Complex (Full-scale site map)",
}


def displayMapName(map_name: str) -> str:
    """Return a presentation-friendly map name."""
    return MAP_DISPLAY_NAMES.get(map_name, map_name)
