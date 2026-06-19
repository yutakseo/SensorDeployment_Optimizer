from __future__ import annotations

MAP_ORDER: tuple[str, ...] = (
    "gangjin.full",
    "gangjin.up",
    "gangjin.down",
    "seocho.full",
    "seocho.up",
    "seocho.down",
    "sejong.full",
)

MAP_DISPLAY_NAMES: dict[str, str] = {
    "gangjin.full": "Highway: full-scale",
    "gangjin.up": "Highway: Upper-region",
    "gangjin.down": "Highway: Lower-region",
    "seocho.full": "Office Building: full-scale",
    "seocho.up": "Office Building: Upper-region",
    "seocho.down": "Office Building: Lower-region",
    "sejong.full": "Industrial Complex: full-scale",
}
DISPLAY_TO_CANONICAL: dict[str, str] = {
    display_name: map_name for map_name, display_name in MAP_DISPLAY_NAMES.items()
}


def displayMapName(map_name: str) -> str:
    """Return a presentation-friendly map name."""
    return MAP_DISPLAY_NAMES.get(map_name, map_name)


def mapOrderKey(map_name: str) -> tuple[int, str]:
    """Return the fixed report order key for a canonical map name."""
    map_name = DISPLAY_TO_CANONICAL.get(map_name, map_name)
    try:
        return (MAP_ORDER.index(map_name), "")
    except ValueError:
        return (len(MAP_ORDER), map_name)


def sortMapNames(map_names: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    """Sort canonical map names in the required report row order."""
    return sorted(map_names, key=mapOrderKey)
